"""Freeze the patient-disjoint validation set: unselected training pairs (cells beyond 60) whose
patients are absent from the 2,000 sampled images, the evaluation sample, the dev slice, and the gold
set. Written as a pairs file in the harness's format so verify_run.py reads it unchanged."""
import sys
import glob, json, hashlib, os
from collections import Counter
RAD=os.environ.get("RADOPEN_ROOT") or os.environ.get("HARNESS_ROOT") or sys.exit("set RADOPEN_ROOT to the harness directory (README, Setup)"); B=os.environ.get("CLAIMBLIND_ROOT", ".")
pf=sorted(glob.glob(RAD+"/results/e6_pairs/pairs_*.jsonl"))[-1]; gf=sorted(glob.glob(RAD+"/results/e6_pairs/gold_pairs_*.jsonl"))[-1]
pairs=[json.loads(l) for l in open(pf) if l.strip()]; gold=[json.loads(l) for l in open(gf) if l.strip()]
def pat(p): return p.split("/")[1]
sel=[l.strip() for l in open(B+"/results/train_slice/selected_round0.txt") if l.strip()]
blocked={pat(s) for s in sel}
cells=Counter(); trn=[]
for p in pairs:
    k=(p["finding"],p["direction"]); i=cells[k]; cells[k]+=1
    if i<60: blocked.add(pat(p["report_path"])); blocked.add(pat(p["image_path"]))
    else: trn.append(p)
for g in gold: blocked.add(pat(g["report_path"])); blocked.add(pat(g["image_path"]))
sel_imgs=set(sel)
clean=[p for p in trn if p["report_path"] not in sel_imgs and p["image_path"] not in sel_imgs and pat(p["report_path"]) not in blocked and pat(p["image_path"]) not in blocked]
# also require the two patients of a pair to be distinct from other clean pairs' patients? keep all; report reuse
pc=Counter(); [pc.update([pat(p["report_path"]),pat(p["image_path"])]) for p in clean]
out=B+"/results/val_clean/pairs_val_clean_2026-09-24.jsonl"
import os; os.makedirs(B+"/results/val_clean",exist_ok=True)
with open(out,"w") as f:
    for p in clean: f.write(json.dumps(p)+"\n")
h=hashlib.sha256(open(out,"rb").read()).hexdigest()
cc=Counter((p["finding"],p["direction"]) for p in clean)
print("VAL_CLEAN pairs",len(clean),"cells",len(cc),dict(cc),"patients",len(pc),"patients_reused",sum(1 for v in pc.values() if v>1),"sha256",h)
imgs={p["report_path"] for p in clean}|{p["image_path"] for p in clean}
with open(B+"/results/val_clean/images.txt","w") as f: f.write("\n".join(sorted(imgs))+"\n")
print("images",len(imgs))
