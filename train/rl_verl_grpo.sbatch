#!/bin/bash
# Cluster settings (partition, QOS, account) are passed on the sbatch command line; --gres names the GPU type used in the runs.
#SBATCH --gres=gpu:a6000:3
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J rl_grpo
#SBATCH -o logs/%x_%j.out
# Online RLVR with verl (GRPO, vLLM rollout, FSDP2 actor, LoRA) on Qwen3-VL-8B-Instruct,
# design v1 (2026-09-24): pair-only verifiable reward computed by a frozen base adjudicator that is
# served by vLLM on the LAST GPU of the allocation; verl uses the other GPUs (TP=1, one engine per
# GPU). Ray runs single-node inside the job. Adapted from verl's examples/grpo_trainer/
# run_qwen3_vl_8b_fsdp.sh (pip install, so VERL_USE_UV=0). Fails loud: the adjudicator must be
# ready before verl starts; a nonzero verl exit fails the job; the checkpoint directory must exist.
# Smoke: sbatch --export=ALL,SMOKE=1 --gres=gpu:<GPU>:2 -c 8 --mem=128G -t 1:30:00 -J rl_grpo_smoke code/rl_verl_grpo.sbatch
# Run:   sbatch --export=ALL,EXP=v1_paironly_seed1,SEED=1 code/rl_verl_grpo.sbatch
set -uo pipefail
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID DATE=$(date) GPUS=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ROOT=${CLAIMBLIND_ROOT:-.}
RAD=${RADOPEN_ROOT:-${HARNESS_ROOT:?set RADOPEN_ROOT to the harness directory (README, Setup)}}
export RADOPEN_ROOT=$RAD
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
SMOKE=${SMOKE:-0}
EXP=${EXP:-smoke_$SLURM_JOB_ID}
SEED=${SEED:-1}
DATA=${DATA:-data/verl_v1}
TRAIN_FILE=$ROOT/$DATA/train.parquet; VAL_FILE=$ROOT/$DATA/val.parquet
[ -s "$TRAIN_FILE" ] && [ -s "$VAL_FILE" ] || { echo "NO_DATA $TRAIN_FILE"; exit 2; }
export HF_HOME=$RAD/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERL_USE_UV=0 WANDB_MODE=disabled
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES   # Slurm exports the ROCm variable too; verl/Ray refuse it next to CUDA_VISIBLE_DEVICES
[ -f ~/.hf_env ] && source ~/.hf_env
source "${VERL_VENV:?set VERL_VENV to the verl environment}/bin/activate"
python3 -c "import verl, vllm, torch; print('verl', verl.__version__, 'vllm', vllm.__version__, 'torch', torch.__version__)" || { echo "ENV_BROKEN"; exit 2; }
# ---- GPU split: last GPU = adjudicator server, the rest = verl ----
ALL_GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
NG=${#ALL_GPUS[@]}
[ "$NG" -ge 2 ] || { echo "NEED_AT_LEAST_2_GPUS"; exit 2; }
ADJ_GPU=${ALL_GPUS[$((NG-1))]}
VERL_GPUS=$(IFS=,; echo "${ALL_GPUS[*]:0:$((NG-1))}")
N_VERL=$((NG-1))
PORT=$((20000 + SLURM_JOB_ID % 20000))
ADJ_MODEL=${JUDGE_MODEL:-$MODEL}   # 2026-09-26: judge may differ from the policy (second writer)
export ADJ_BASE_URL="http://127.0.0.1:${PORT}/v1" ADJ_MODEL
mkdir -p $ROOT/logs $ROOT/runs/verl
CUDA_VISIBLE_DEVICES=$ADJ_GPU python3 -m vllm.entrypoints.openai.api_server --model "$ADJ_MODEL" --port "$PORT" --max-model-len 8192 \
  --dtype bfloat16 --seed 20260919 --gpu-memory-utilization 0.80 --limit-mm-per-prompt '{"image":0}' > $ROOT/logs/adj_${SLURM_JOB_ID}.log 2>&1 &
APID=$!
trap 'kill $APID 2>/dev/null; ray stop --force >/dev/null 2>&1; true' EXIT
for i in $(seq 1 240); do
  curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "\"id\":\"$ADJ_MODEL\"" && { echo "ADJ_READY after $((i*10))s on GPU $ADJ_GPU"; break; }
  kill -0 $APID 2>/dev/null || { echo "ADJ_DIED"; tail -40 $ROOT/logs/adj_${SLURM_JOB_ID}.log; exit 1; }
  sleep 10
done
curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "\"id\":\"$ADJ_MODEL\"" || { echo "ADJ_TIMEOUT"; exit 1; }
# reward function self-test against the live adjudicator (fails loud before any GPU minute is spent on verl)
python3 - <<'PY' || { echo "REWARD_SELFTEST_FAILED"; exit 3; }
import json, os, sys
sys.path.insert(0, os.path.join(os.environ.get("CLAIMBLIND_ROOT", "."), "code"))
import rl_verl_reward as R
rec = {"record": [{"finding": "Pleural Effusion", "present": True, "side": "left", "box": None, "note": "blunted left costophrenic angle"}], "other": ""}
gt = json.dumps({"claims": [{"claim": "There is a small left pleural effusion.", "truth": "supported"}]})
s = R.compute_score("claimblind_cxr", json.dumps(rec), gt)
print("SELFTEST score", s)
assert s in (0.0, 1.0)
PY
# ---- verl ----
if [ "$SMOKE" = "1" ]; then
  TRAIN_BS=8; MINI_BS=8; ROLL_N=4; TOTAL_STEPS=2; SAVE_FREQ=1; TEST_FREQ=1
else
  TRAIN_BS=${TRAIN_BS:-32}; MINI_BS=${MINI_BS:-16}; ROLL_N=${ROLL_N:-8}; TOTAL_STEPS=${TOTAL_STEPS:-0}; SAVE_FREQ=${SAVE_FREQ:-10}; TEST_FREQ=${TEST_FREQ:-10}
fi
STEPS_ARG=""; [ "$TOTAL_STEPS" != "0" ] && STEPS_ARG="trainer.total_training_steps=$TOTAL_STEPS"
# fail fast on a data-parallel split verl rejects at its first update (observed: mini batch
# 16 x 8 = 128 sequences on 3 verl GPUs -> AssertionError after 20 min of init)
for BS in $TRAIN_BS $MINI_BS; do
  [ $(( BS * ROLL_N % N_VERL )) -eq 0 ] || { echo "BATCH_NOT_DIVISIBLE bs=$BS n=$ROLL_N verl_gpus=$N_VERL"; exit 2; }
done
# flash-attn is optional: without it, padding removal and fused kernels are off (slower, same objective)
if python3 -c "import flash_attn" 2>/dev/null; then FLASH_ARGS="actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.use_fused_kernels=True"; echo FLASH_ATTN=yes;
else FLASH_ARGS="actor_rollout_ref.model.use_remove_padding=False +actor_rollout_ref.model.override_config={attn_implementation:sdpa}"; echo FLASH_ATTN=no; fi
CKPT=$ROOT/runs/verl/$EXP
echo "VERL_EXTRA=${VERL_EXTRA:-} ROLL_TEMP=${ROLL_TEMP:-0.8} REWARD_PATH=${REWARD_PATH:-$ROOT/code/rl_verl_reward.py} KEEP_LAST=${KEEP_LAST:-0} LR=${ACTOR_LR:-1e-5} REWARD_LOG_DIR=${REWARD_LOG_DIR:-}"
echo "EXP=$EXP SEED=$SEED VERL_GPUS=$VERL_GPUS N_VERL=$N_VERL TRAIN_BS=$TRAIN_BS MINI_BS=$MINI_BS ROLL_N=$ROLL_N TOTAL_STEPS=$TOTAL_STEPS CKPT=$CKPT"
cd $ROOT
CUDA_VISIBLE_DEVICES=$VERL_GPUS python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=$TRAIN_FILE \
  data.val_files=$VAL_FILE \
  data.image_key=images \
  data.train_batch_size=$TRAIN_BS \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.seed=$SEED \
  actor_rollout_ref.model.path="$MODEL" \
  $FLASH_ARGS \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]' \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-5} \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$N_VERL \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=$ROLL_N \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.temperature=${ROLL_TEMP:-0.8} \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path=${REWARD_PATH:-$ROOT/code/rl_verl_reward.py} \
  reward.custom_reward_function.name=${REWARD_FN:-compute_score} \
  trainer.balance_batch=True \
  trainer.logger='["console"]' \
  trainer.project_name=claimblind_rlvr \
  trainer.experiment_name=$EXP \
  trainer.default_local_dir=$CKPT \
  trainer.n_gpus_per_node=$N_VERL \
  trainer.nnodes=1 \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_epochs=1 \
  ${VERL_EXTRA:-} \
  $STEPS_ARG \
  ray_kwargs.ray_init.runtime_env.py_executable=null 2>&1 | tee $ROOT/logs/verl_${EXP}_${SLURM_JOB_ID}.log
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "VERL_FAILED rc=$rc"; exit 4; }
ls -d $CKPT/global_step_* 2>/dev/null | wc -l | xargs echo "CHECKPOINTS="
[ -n "$(ls -d $CKPT/global_step_* 2>/dev/null)" ] || { echo "NO_CHECKPOINTS"; exit 5; }
# convert every checkpoint to a PEFT adapter inside this allocation (the login node OOM-kills the merger), then prune the FSDP shards
MERGE_FAIL=0
for D in $(ls -d $CKPT/global_step_* | sort -t_ -k3 -n); do
  if [ -f "$D/hf/lora_adapter/adapter_config.json" ]; then continue; fi
  CUDA_VISIBLE_DEVICES="" python3 -m verl.model_merger merge --backend fsdp --local_dir $D/actor --target_dir $D/hf > $D/merge.log 2>&1
  if [ -f "$D/hf/lora_adapter/adapter_config.json" ] || [ -f "$D/hf/adapter_config.json" ]; then
    if [ "${KEEP_LAST:-0}" = "1" ] && [ "$(basename $D)" = "global_step_$(cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null)" ]; then
      rm -f $D/hf/model.safetensors $D/hf/model-*.safetensors; echo "MERGED_KEPT_RESUMABLE $D"   # KEEP_LAST=1: newest save stays resumable (dose-test extension)
    else
      rm -f $D/actor/model_world_size_*.pt $D/actor/optim_world_size_*.pt $D/data.pt $D/hf/model.safetensors $D/hf/model-*.safetensors; echo "MERGED_AND_PRUNED $D"
    fi
  else echo "MERGE_FAILED $D (shards kept)"; MERGE_FAIL=$((MERGE_FAIL+1)); fi
done
find $CKPT -maxdepth 4 -name "adapter_config.json" | head -3
[ "$MERGE_FAIL" -eq 0 ] || { echo "SOME_MERGES_FAILED=$MERGE_FAIL"; exit 6; }
echo "GRPO_EXIT rc=0 EXP=$EXP $(date)"
