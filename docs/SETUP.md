# Setup

[Back to the README](../README.md)

## Environments

Linux with Slurm and NVIDIA GPUs. Two Python 3.11.16 environments, pinned in the requirements files:

* `requirements/verl.txt` (`VERL_VENV`): training with verl 0.9.1, vLLM 0.24.0, PyTorch 2.11.0, Transformers 5.9.0,
  PEFT 0.21.0, Ray 2.58.0; also the parquet builders, the checkpoint merger, the direct-supervision baseline, and the
  CPU statistics.
* `requirements/harness.txt` (`HARNESS_VENV`): checklist writing and every checker read through a local
  OpenAI-compatible vLLM 0.29.0 server (PyTorch 2.13.0, Transformers 5.17.0, openai 3.19.0). The client refuses
  non-local endpoints.
* `requirements/figures.txt`: matplotlib, NumPy, and Pillow for the figure builders.

The versions were read from the environments that ran the study.

## Working directory and environment variables

```bash
export CLAIMBLIND_ROOT=/abs/path/workdir        # will hold code/, data/, results/, runs/, logs/, domain2/
bash setup_code_dir.sh                          # copies every script and config into $CLAIMBLIND_ROOT/code/
export RADOPEN_ROOT=/abs/path/to/repo/harness   # required: the harness and its data (below)
export VERL_VENV=/path/to/verl-env HARNESS_VENV=/path/to/harness-env
export PARTITION=<partition> QOS=<qos>          # used by the launchers
cd $CLAIMBLIND_ROOT                             # submit every job from here (logs go to logs/%x_%j.out)
```

`RADOPEN_ROOT` must be set. The study's scripts defaulted to the harness path on the cluster where they were
developed; the released scripts have no such default and stop with a message when it is unset (`HARNESS_ROOT` is
accepted as another name for it). Besides `code/`, the harness directory must hold `datasets/chexpert_plus/` (the
dataset files, see `pairs/README.md`), `results/e6_pairs/` (the pair build and the `image_index.tsv` written by
`harness/code/scripts/index_images.py`), `runs/image_cache/` (1024-pixel PNGs written on first use), and `hf_cache/`
(used as `HF_HOME`; every job runs offline). Set both roots to absolute paths. Relative paths inside the configs
(`results/...`, `data/...`, `runs/...`) resolve under `$CLAIMBLIND_ROOT`. Cluster settings (partition, QOS, account)
are passed on the sbatch command line; the `--gres` lines name the GPU types the runs used.

| variable | meaning |
|---|---|
| `CLAIMBLIND_ROOT` | working directory for code, data, results, runs, logs (default `.`; set it to an absolute path) |
| `RADOPEN_ROOT` | harness directory, required (`HARNESS_ROOT` is accepted as another name) |
| `VERL_VENV`, `HARNESS_VENV` | the two Python environments |
| `PARTITION`, `QOS` | Slurm partition and QOS used by the launchers |
| `VQA_ROOT` | photograph study: directory with the VQA v2 JSON files and the COCO `train2014/` and `val2014/` images; the COCO instance annotations go in `$CLAIMBLIND_ROOT/domain2/data/annotations/` |
| `FIG_ROOT` | figure builders: directory with `artifacts/` (inputs) and `figs/` (outputs) |
| `ADJ_BASE_URL`, `ADJ_MODEL` | set by `train/rl_verl_grpo.sbatch`: the local vLLM server of the training checker |
| `WRITER_MODEL` | set by the read jobs: `record` routes the checklist model to the served adapter, `BASE` to the base model |
| `REWARD_LOG_DIR` | per-claim reward log directory (logged and omission-cost rewards) |
| `JUDGE_CONVENTION` | optional sentence appended to the checker message (the stated-convention rereads); unset leaves the message unchanged |
| `JUDGE_PROMPT_VARIANT` | optional `cite` or `reason` reply line (the checker-instruction rereads); unset leaves the message unchanged |
| `VLLM_EXTRA` | extra vLLM server arguments for a reread (for example `--tensor-parallel-size 2`) |

## Models and revisions

Every job loads models by Hugging Face id from an offline cache that held one snapshot per model, at these revisions:

| model | role | revision |
|---|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` | checklist model (with LoRA) and training checker | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| `google/medgemma-4b-it` | independent checker | `290cda5eeccbee130f987c4ad74a59ae6f196408` |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | further checker | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| `microsoft/Phi-4-mini-instruct` | further checker | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `allenai/OLMo-2-1124-7B-Instruct` | further checker | `470b1fba1ae01581f270116362ee4aa1b97f4c84` |
| `google/gemma-3-4b-it` | checker size series | `093f9f388b31de276ce2de164bdc2081324b9767` |
| `google/gemma-3-12b-it` | checker size series; photograph study checker | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| `google/gemma-3-27b-it` | checker size series | `005ad3404e59d6023443cb575daa05336842228a` |
| `Qwen/Qwen2.5-VL-7B-Instruct` | second checklist model; photograph study checklist model | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `HuggingFaceM4/Idefics3-8B-Llama3` | photograph study screen | `fddb4ff79181e55a994674777e06cd5456ce3dc3` |

Download each into `$RADOPEN_ROOT/hf_cache` at its revision, for example
`HF_HOME=$RADOPEN_ROOT/hf_cache huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
The Gemma and MedGemma models require accepting their terms on Hugging Face. The photograph study passes the pinned
revisions to vLLM itself (`photo_study/model_revisions.txt`).

The trained adapters are not released. For reference, the sha256 of `adapter_model.safetensors` at step 40 of the three
replication runs: seed 101 `d1d4116404d9b9f42ee0ee96358d6cdc8082b4b2c6e28efbe81395d6f7afe634`, seed 202
`eebf8f3ce37f91768b349349452ae30226d300ae83dbf76c13a12015496a2909`, seed 303
`f7bbf6840b6afb8eb0d5c57ef07044e1e54fa880a68947f5671d34b482cd6d09`.
