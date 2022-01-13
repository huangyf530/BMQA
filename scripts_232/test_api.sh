#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

WORKING_DIR=/data1/private/hyf/BMQA
CONFIG_PATH="${WORKING_DIR}/configs/model/t5_large_config.json"
CKPT_PATH="${WORKING_DIR}/deploy_model/unified_large_5000_0.538"

MASTER_ADDR=localhost
MASTER_PORT=8888
NNODES=1
NODE_RANK=0

GPUS_PER_NODE=1
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

MP_SIZE=1
SEED=2021
SAVE_PATH="${WORKING_DIR}/results/unified/fintune/lr${LR}_B${GLOBAL_BATCH_SIZE}_finetune_seed${SEED}_large"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model_fp32.json"
TOKENIZER_PATH="${WORKING_DIR}/sp_t5"

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --no-load-optim"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/test.py ${OPTS}"

echo ${CMD}
${CMD}
