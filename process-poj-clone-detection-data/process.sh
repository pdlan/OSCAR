#!/bin/bash
max_len=511
in_dir=../data-raw/poj-clone-detection
out_dir=../data-bin/poj-clone-detection
tmp_dir=./out
moco_path=../data-bin/pretrain
for split in train valid test
do
    mkdir -p $tmp_dir/$split/source_code $tmp_dir/$split/llvm_ir $tmp_dir/$split/json \
        $tmp_dir/$split/concat_json $tmp_dir/$split/rawtext $tmp_dir/result/$split
    python3 1_extract_source_code.py $in_dir/$split.jsonl $tmp_dir/$split/source_code
    python3 2_generate_makefile.py $tmp_dir/$split/source_code $tmp_dir/$split/llvm_ir $tmp_dir/$split/Makefile
    make -j24 -k -f $tmp_dir/$split/Makefile
    python3 3_ir_to_json.py $tmp_dir/$split/llvm_ir $tmp_dir/$split/json $max_len
    python3 4_concat_json.py $tmp_dir/$split/llvm_ir $tmp_dir/$split/json \
        $tmp_dir/$split/concat_json funclist.json $in_dir/$split.jsonl label.txt
    python3 5_json_to_rawtext.py $tmp_dir/$split/concat_json $tmp_dir/$split/rawtext \
        $moco_path/inst_dict.txt $moco_path/state_dict.txt $max_len
    cp $tmp_dir/$split/rawtext/* $tmp_dir/result/$split/
    cp $tmp_dir/$split/concat_json/label.txt $tmp_dir/result/$split/
done
cd ../model
./scripts/poj_clone_detection_preprocess.sh ../process-poj-clone-detection-data/out/result/ $out_dir ../data-bin/pretrain
