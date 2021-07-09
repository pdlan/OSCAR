#!/bin/bash
CHECKPOINT=$1
DATA_DIR=../data-bin/bindiff/
SAVE_DIR=../checkpoint/bindiff/
TOTAL_UPDATES=128000
WARMUP_UPDATES=6400
PEAK_LR=0.00002
MAX_POSITIONS=511
MAX_SENTENCES=2
UPDATE_FREQ=1
SEED=65535
ENCODER_LAYERS=6
SMALLBERT_ENCODER_LAYERS=3
SMALLBERT_INSTS_PER_GROUP=4
VARIANTS=4
FP16=--fp16
STATE=
POOLING=
python setup.py build_ext --inplace
NLABELS=$(( `cat $DATA_DIR/label/train.txt|wc -l` / VARIANTS ))
python train.py --num-labels $NLABELS $FP16 \
    --encoder-normalize-before $DATA_DIR --num-workers 0 \
    --moco-queue-length 65536 --moco-projection-dim 256 --moco-temperature 0.05 --moco-momentum 0.999 --ddp-backend=c10d \
    --task poj_similarity --criterion poj_similarity --arch irbert_base \
    --function-length $MAX_POSITIONS \
    --encoder-layers $ENCODER_LAYERS --smallbert-insts-per-group $SMALLBERT_INSTS_PER_GROUP \
    --smallbert-num-encoder-layers $SMALLBERT_ENCODER_LAYERS --smallbert-num-attention-heads 12 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --seed $SEED \
    --embedding-normalize \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
    --find-unused-parameters $STATE $POOLING \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR --restore-file $CHECKPOINT