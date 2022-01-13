#! /bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
WORKING_DIR=/mnt/sfs_turbo/hyf/BMQA

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTIONS_NCCL="NCCL_DEBUG=info"

MP_SIZE=2
DATA_PARALLEL_SIZE=`expr $GPUS_PER_NODE / $MP_SIZE \* $DLS_TASK_NUMBER`

DATA_EXT=".jsonl"
DATA_PATH="/mnt/sfs_turbo/hyf/data_en/QADataset/NewsQA"

LR=${1-0.000005}
GRAD_ACC=${2-1}
SEED=${3-1234}

BATCH_SIZE=4
EVAL_BATCH_SIZE=16
TRAIN_ITER=-1
EPOCHS=5

GLOBAL_BATCH_SIZE=`expr $BATCH_SIZE \* $DATA_PARALLEL_SIZE \* $GRAD_ACC`

CONFIG_PATH="${WORKING_DIR}/configs/model/t5_xl_config.json"
CKPT_PATH="/mnt/sfs_turbo/gyx/checkpoints/t5-xl-2/t5-MP2/"

SAVE_PATH="${WORKING_DIR}/results/newsqa/fintune/lr${LR}_B${GLOBAL_BATCH_SIZE}_finetune_seed${SEED}_xl"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model_fp32.json"
TOKENIZER_PATH="${WORKING_DIR}/sp_t5"


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name unifiedqa"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --dropout 0.1"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style linear"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.05"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 1000000"
OPTS+=" --eval-interval 500"
OPTS+=" --max-save 1"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
# OPTS+=" --fp16"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --seed ${SEED}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --tensorboard ${SAVE_PATH}/tensorboard"
OPTS+=" --dec-seq-length 256"

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
