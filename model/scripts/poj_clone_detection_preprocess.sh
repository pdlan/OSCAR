#!/bin/bash
data_dir=$1
bin_dir=$2
moco_path=$3
mkdir -p $bin_dir/ $bin_dir/states $bin_dir/states $bin_dir/pos $bin_dir/label
python3 preprocess.py --srcdict $moco_path/inst_dict.txt --only-source --trainpref $data_dir/train/insts.txt --validpref $data_dir/valid/insts.txt --testpref $data_dir/test/insts.txt --destdir $bin_dir/insts --dataset-impl mmap --workers 60
python3 preprocess.py --srcdict $moco_path/state_dict.txt --only-source --trainpref $data_dir/train/states.txt --validpref $data_dir/valid/states.txt --testpref $data_dir/test/states.txt --destdir $bin_dir/states --dataset-impl mmap --workers 60
python3 preprocess_pos.py poj $data_dir/ $bin_dir/pos/
cp $data_dir/train/label.txt $bin_dir/label/train.txt
cp $data_dir/valid/label.txt $bin_dir/label/valid.txt
cp $data_dir/test/label.txt $bin_dir/label/test.txt
cp $moco_path/inst_dict.txt $bin_dir/
cp $moco_path/state_dict.txt $bin_dir/
