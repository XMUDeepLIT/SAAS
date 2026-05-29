#!/bin/bash
set -ex

#timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
timestamp=$(date -d "+8 hour" "+%Y-%m-%d-%H:%M")
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
BASE_DIR=${BASE_DIR:-"root"}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

data_dir=/the/path/you/save/data
train_data=${data_dir}/train.jsonl
val_data=${data_dir}/test.jsonl

model_dir=/the/path/you/save/model
train_model_dir=/the/path/you/save/model_megatron
model_name=your_model_name

origin_model_path=${model_dir}/${model_name}
train_model_path=${train_model_dir}/${model_name}


# gpu parameters
tp=1
N_GPUS=4
gpu_memory_utilization=0.8

# rollout parameters
n_rollout=8
rollout_temperature=1.0
rollout_top_k=-1 

# train parameters
train_batch_size=512
max_prompt_length=512
max_response_length=512
lr=1e-6
lr_warmup_steps_ratio=0.285

# validate parameters
save_freq=50
test_freq=50
val_n=1
val_temperature=1.0
val_top_k=-1 # -1 for vLLM rollout, 0 for HF rollout.
val_top_p=1.0

# tool parameters
max_query_nums=6
query_rep_penalty=True

efficient_query=True

disable_search_tokens_for_first_half=True
disable_search_first_n=4

# experiment_name=eval-3b-adaptive-maj1
experiment_name=your_experiment_name

data_saved_path_dir=/the/path/you/save/data_saved
data_saved_path=${data_saved_path_dir}/${experiment_name}
reward_info_path_dir=$data_saved_path/reward_info
ckpts_path_dir=$data_saved_path/ckpts

# log and tb
log_path=$data_saved_path/log/logs-${timestamp}.log
tb_path_dir=$data_saved_path_dir/tb_logs
WANDB_KEY=your_wandb_key

mkdir -p $reward_info_path_dir $ckpts_path_dir $data_saved_path/log0
tb_path=$tb_path_dir/${experiment_name}
export TENSORBOARD_DIR=$tb_path

exec > >(tee -a "${log_path}") 2>&1
cp $0  $data_saved_path

CKPT_ARGS=(
   --hf-checkpoint ${origin_model_path}
   --ref-load ${train_model_path}
   --load ${ckpts_path_dir}
   --save ${ckpts_path_dir}
   --save-interval ${save_freq}
)

ROLLOUT_ARGS=(
   # --num-epoch 1
   --num-rollout 1
   --prompt-data ${train_data}
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   # --rollout-shuffle
   --rollout-batch-size ${train_batch_size}
   --n-samples-per-prompt ${n_rollout}
   --rollout-max-response-len ${max_response_length}
   --rollout-temperature ${rollout_temperature}
   --rollout-max-prompt-len ${max_prompt_length}
   --use-slime-router

   # eval args
   --eval-interval ${test_freq}
   --eval-prompt-data nq_test ${val_data}@[0:3610] triviaqa_test ${val_data}@[3610:14923] popqa_test ${val_data}@[14923:29190] hotpotqa_test ${val_data}@[29190:36595] 2wiki_test ${val_data}@[36595:49171] musique_test ${val_data}@[49171:51588] bamboogle_test ${val_data}@[51588:51713]
   # --eval-prompt-data nq_test ${val_data}
   --eval-input-key prompt
   --eval-label-key reward_model
   --n-samples-per-eval-prompt ${val_n}
   --eval-temperature ${val_temperature}
   --eval-top-p ${val_top_p}
   --eval-top-k ${val_top_k} 
   --eval-max-prompt-len ${max_prompt_length}

   --global-batch-size $(( train_batch_size * n_rollout ))
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size ${tp}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${lr}
   --lr-warmup-fraction ${lr_warmup_steps_ratio}
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --no-check-for-nan-in-loss-and-grad
)

WANDB_ARGS=(
   # --use-wandb
   --wandb-project slime-search
   --wandb-group ${experiment_name}
   --wandb-key ${WANDB_KEY}
)

TENSORBOARD_ARGS=(
   --use-tensorboard
   --tb-project-name ${tb_path_dir}
   --tb-experiment-name ${experiment_name}
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${tp}
   --sglang-mem-fraction-static ${gpu_memory_utilization}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path search.generate_with_search.generate
   --rollout-function-path search.sglang_rollout.generate_rollout
   --custom-mq-rm-path search.generate_with_search.mq_group_reward_func
   --custom-nq-rm-path search.generate_with_search.mq_group_reward_func
   --custom-eval-rm-path search.generate_with_search.eval_reward_func
   --group-rm

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
   --reward-info-path ${reward_info_path_dir}
   --lap-ratio-threshold ${lap_ratio_threshold}
   --max-query-nums ${max_query_nums}
   --disable-search-first-n ${disable_search_first_n}
)

# Conditionally add store_true type arguments
[ "$efficient_query" = "True" ] && CUSTOM_ARGS+=(--efficient-query)
[ "$doc_rep_penalty" = "True" ] && CUSTOM_ARGS+=(--doc-rep-penalty)
[ "$query_rep_penalty" = "True" ] && CUSTOM_ARGS+=(--query-rep-penalty)
[ "$format_check" = "True" ] && CUSTOM_ARGS+=(--format-check)
[ "$rsp_rep_detect" = "True" ] && CUSTOM_ARGS+=(--repetition-detect)
[ "$use_linear_fn" = "True" ] && CUSTOM_ARGS+=(--linear-fn)
[ "$retrieval_penalty" = "True" ] && CUSTOM_ARGS+=(--retrieval-penalty)
[ "$retrieval_em_different" = "True" ] && CUSTOM_ARGS+=(--retrieval-em-different)
[ "$disable_search_tokens_for_first_half" = "True" ] && CUSTOM_ARGS+=(--disable-search-tokens-for-first-half)

# Extract the custom generate function path and copy it along with rollout.py to data_saved_path
CUSTOM_GENERATE_FILE=$(echo generate_with_search | tr '.' '/').py
cp ${SCRIPT_DIR}/../${CUSTOM_GENERATE_FILE} ${data_saved_path}/
cp ${SCRIPT_DIR}/../../slime/ray/rollout.py ${data_saved_path}/


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

export PYTHONPATH=/root/Megatron-LM

python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${N_GPUS} \
   --rollout-num-gpus ${N_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

