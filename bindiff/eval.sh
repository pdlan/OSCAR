#!/bin/bash
MAX_POSITIONS=511
ENCODER_LAYERS=6
SMALLBERT_ENCODER_LAYERS=3
SMALLBERT_INSTS_PER_GROUP=4
FP16=--fp16
STATE=
POOLING=
CHECKPOINT=$1
PROGRAM=$2
TMP_FEATURES_DIR=/dev/shm/`uuidgen`
cd ../model
mkdir -p $TMP_FEATURES_DIR

for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i python generate_bindiff_features.py \
        --function-length $MAX_POSITIONS --encoder-layers $ENCODER_LAYERS \
        --smallbert-encoder-layers $SMALLBERT_ENCODER_LAYERS \
        --smallbert-insts-per-group $SMALLBERT_INSTS_PER_GROUP \
        --smallbert-num-attention-heads 12 \
        --checkpoint $CHECKPOINT --data ../data-bin/bindiff-eval/$PROGRAM/O$i-gcc7.5.0-amd64 \
        --output $TMP_FEATURES_DIR/O$i-gcc7.5.0-amd64.pt \
        $FP16 $STATE $POOLING &
    python eval.py $PROGRAM $TMP_FEATURES_DIR
done
wait
rm -rf $TMP_FEATURES_DIR
