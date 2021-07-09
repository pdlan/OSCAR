#!/bin/bash
DATA_DIR=../data-bin/pretrain
SAVE_DIR=../checkpoint/pretrain
TOTAL_UPDATES=1000000
WARMUP_UPDATES=30000
PEAK_LR=0.0001
MAX_POSITIONS=511
MAX_SENTENCES=1
UPDATE_FREQ=4
SEED=65535
ENCODER_LAYERS=6
SMALLBERT_ENCODER_LAYERS=3
SMALLBERT_INSTS_PER_GROUP=4
VARAINTS=20
MOCO_QUEUE_LENGTH=65536
MOCO_PROJECTION_DIM=256
MOCO_TEMPERATURE=0.05
MOCO_MOMENTUM=0.999
MOCO_LOSS_COEFF=1000
FP16=--fp16
STATE=
POOLING=
python setup.py build_ext --inplace
python train.py --encoder-normalize-before $FP16 $DATA_DIR --num-workers 0 \
    --encoder-layers $ENCODER_LAYERS --smallbert-insts-per-group $SMALLBERT_INSTS_PER_GROUP \
    --smallbert-num-encoder-layers $SMALLBERT_ENCODER_LAYERS --smallbert-num-attention-heads 12 \
	--moco-valid-size 500 --moco-queue-length $MOCO_QUEUE_LENGTH --moco-projection-dim $MOCO_PROJECTION_DIM \
    --moco-temperature $MOCO_TEMPERATURE --moco-momentum $MOCO_MOMENTUM --ddp-backend=c10d \
	--task ir_masked_lm --criterion ir_masked_lm --arch irbert_base --augmented-variants $VARAINTS \
    --mlm-loss-coeff 1 --moco-loss-coeff $MOCO_LOSS_COEFF \
	--function-length $MAX_POSITIONS \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 1.0 \
	--lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --seed $SEED \
	--mask-prob 0.15 $STATE $POOLING \
	--embedding-normalize --find-unused-parameters \
	--max-update $TOTAL_UPDATES --log-format simple --log-interval 1000 \
	--skip-invalid-size-inputs-valid-test \
	--save-dir $SAVE_DIR
