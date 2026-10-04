#!/usr/bin/env python3
"""Direct-supervision baseline (2026-09-26; protocol declared before any run).

The same writer (Qwen3-VL-8B-Instruct) and the same LoRA (r 16, alpha 32, q/k/v/o/gate/up/down of the language
model; the vision tower is untouched) as the GRPO runs, trained by supervised next-token loss to write the record
that the image's own report labels imply: twelve entries in the fixed finding order, present = label 1, side
"not applicable", box null, note "", other "" (the oracle_labels record format). The prompt is the training
parquet's writer prompt, unchanged, so the adapter is read by the unchanged evaluation-v2 pipeline.
Modes:
  full    every token of the record is supervised (all twelve flags from the label vector)
  target  only the entries of the findings named by the image's training claims are supervised: the same
          per-image information the matched-pair reward carries, given directly instead of through a judge
Image sets: --paths FILE (one image path per line, e.g. the images the discovery RL run scored) or all rows.
One epoch, micro-batch 1, gradient accumulation 8, AdamW LR 1e-4, 3 percent linear warmup then cosine, clip 1.0,
seed 101. Saves a PEFT adapter whose tensor names must equal the GRPO adapters' (checked; exit 3 otherwise).
"""
import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

FINDINGS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def label_record(labels):
    return {"record": [{"finding": f, "present": bool(x == 1), "side": "not applicable", "box": None, "note": ""}
                       for f, x in zip(FINDINGS, labels)], "other": ""}


def entry_spans(text, rec, keep):
    """Character spans of the entries whose finding is in keep, located in the serialized record."""
    spans, pos = [], 0
    for e in rec["record"]:
        s = json.dumps(e)
        i = text.index(s, pos); pos = i + len(s)
        if e["finding"] in keep:
            spans.append((i, pos))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training parquet (data/rep_s101/train.parquet)")
    ap.add_argument("--paths", default="", help="optional file of image paths to keep")
    ap.add_argument("--mode", choices=("full", "target", "flags_weighted"), required=True)
    ap.add_argument("--pos-weight", type=float, default=6.35, help="flags_weighted: weight of a positive flag token")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-adapter", required=True, help="a GRPO lora_adapter dir whose tensor names must match")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N examples")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    t = pq.read_table(args.data)
    rows = t.to_pylist()
    # the label vector's order is the writer prompt's finding list; refuse to train if they ever differ
    listed = rows[0]["prompt"][1]["content"].split("Findings to cover: ", 1)[1].split(".\n", 1)[0].split(", ")
    if listed != FINDINGS:
        print("FINDINGS_ORDER_MISMATCH", listed); sys.exit(2)
    if args.paths:
        keep = {l.strip() for l in open(args.paths, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["extra_info"]["path"] in keep]
        missing = len(keep) - len(rows)
        print("PATHS keep=%d matched=%d missing=%d" % (len(keep), len(rows), missing), flush=True)
        if missing:
            print("PATHS_MISSING", missing); sys.exit(2)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    if args.limit:
        order = order[:args.limit]
    print("EXAMPLES %d mode=%s epochs=%d" % (len(order), args.mode, args.epochs), flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    vis_lora = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
    if vis_lora:
        print("VISION_LORA_PRESENT", vis_lora[:3]); sys.exit(3)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = math.ceil(len(order) * args.epochs / args.accum)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    end_ids = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    model.train(); step = 0; micro = 0; t0 = time.time(); run_loss = 0.0; run_tok = 0; sup_tokens = 0
    for ep in range(args.epochs):
        for k, i in enumerate(order):
            r = rows[i]
            sys_msg, user_msg = r["prompt"][0]["content"], r["prompt"][1]["content"]
            img = Image.open(io.BytesIO(r["images"][0]["bytes"])).convert("RGB")
            msgs = [{"role": "system", "content": [{"type": "text", "text": sys_msg}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_msg.replace("<image>", "")}]}]
            ptxt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = proc(text=[ptxt], images=[img], return_tensors="pt")
            rec = label_record(r["extra_info"]["labels"])
            target = json.dumps(rec)
            te = tok(target, add_special_tokens=False, return_offsets_mapping=True)
            tids = te["input_ids"]
            wts = None
            if args.mode == "full":
                tlab = list(tids) + end_ids
            elif args.mode == "flags_weighted":
                # 2026-09-26 conditional arm: supervise ONLY the twelve present-value words, positives weighted
                vspans, pos = [], []
                pos_c = 0
                for e in rec["record"]:
                    s_ = json.dumps(e); i0 = target.index(s_, pos_c); pos_c = i0 + len(s_)
                    j = target.index('"present": ', i0) + len('"present": ')
                    word = "true" if e["present"] else "false"
                    vspans.append((j, j + len(word))); pos.append(e["present"])
                tlab, wts = [], []
                for tid, (a, b) in zip(tids, te["offset_mapping"]):
                    hit = [k for k, (s0, e0) in enumerate(vspans) if a < e0 and b > s0]
                    tlab.append(tid if hit else -100)
                    wts.append((args.pos_weight if pos[hit[0]] else 1.0) if hit else 0.0)
                tlab += [-100] * len(end_ids); wts += [0.0] * len(end_ids)
            else:
                gt = json.loads(r["reward_model"]["ground_truth"])
                keep = {c["finding"] for c in gt["claims"] if c.get("finding") in FINDINGS}
                spans = entry_spans(target, rec, keep)
                tlab = [tid if any(a < e and b > s for s, e in spans) else -100 for tid, (a, b) in zip(tids, te["offset_mapping"])]
                tlab += [-100] * len(end_ids)
            ids = torch.cat([enc["input_ids"][0], torch.tensor(tids + end_ids)]).unsqueeze(0).cuda()
            lab = torch.cat([torch.full_like(enc["input_ids"][0], -100), torch.tensor(tlab)]).unsqueeze(0).cuda()
            extra = {}
            if "mm_token_type_ids" in enc:   # transformers >= 5: multimodal RoPE needs the token types; appended text = 0
                mm = enc["mm_token_type_ids"][0]
                extra["mm_token_type_ids"] = torch.cat([mm, torch.zeros(len(tids) + len(end_ids), dtype=mm.dtype)]).unsqueeze(0).cuda()
            nsup = int((lab != -100).sum())
            if nsup == 0:
                continue
            sup_tokens += nsup
            if wts is None:
                out_ = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lab,
                             pixel_values=enc["pixel_values"].cuda(), image_grid_thw=enc["image_grid_thw"].cuda(), **extra)
                loss = out_.loss
            else:
                out_ = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             pixel_values=enc["pixel_values"].cuda(), image_grid_thw=enc["image_grid_thw"].cuda(), **extra)
                w = torch.cat([torch.zeros(enc["input_ids"].shape[1]), torch.tensor(wts)]).cuda()
                logits = out_.logits[0, :-1].float(); tgt = lab[0, 1:]; ww = w[1:]
                m = tgt != -100
                ce = torch.nn.functional.cross_entropy(logits[m], tgt[m], reduction="none")
                loss = (ce * ww[m]).sum() / ww[m].sum()
            (loss / args.accum).backward()
            run_loss += float(loss) * nsup; run_tok += nsup; micro += 1
            if micro % args.accum == 0 or k == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 or step == 1 or step == total:
                    print("STEP %d/%d loss=%.4f lr=%.2e sup_tokens=%d elapsed=%.0fs" % (step, total, run_loss / max(1, run_tok), sched.get_last_lr()[0], sup_tokens, time.time() - t0), flush=True)
                    run_loss = 0.0; run_tok = 0

    ad = out / "lora_adapter"
    model.save_pretrained(str(ad))
    from safetensors import safe_open
    mine = set(safe_open(str(ad / "adapter_model.safetensors"), "pt").keys())
    ref = set(safe_open(str(Path(args.ref_adapter) / "adapter_model.safetensors"), "pt").keys())
    same = mine == ref
    meta = {"mode": args.mode, "examples": len(order), "epochs": args.epochs, "optimizer_steps": step, "lr": args.lr,
            "accum": args.accum, "seed": args.seed, "supervised_tokens": sup_tokens, "seconds": round(time.time() - t0, 1),
            "data": args.data, "paths": args.paths,
            "paths_sha256": hashlib.sha256(open(args.paths, "rb").read()).hexdigest() if args.paths else None,
            "adapter_keys": len(mine), "ref_keys": len(ref), "keys_match_ref": same}
    (out / "sft_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("META", json.dumps(meta), flush=True)
    if not same:
        print("ADAPTER_KEYS_MISMATCH", sorted(mine ^ ref)[:5]); sys.exit(3)
    print("SFT_DONE", out, flush=True)


if __name__ == "__main__":
    main()
