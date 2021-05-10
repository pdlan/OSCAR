#!/bin/bash
data_dir=$1
bin_dir=$2
variants=$3
for i in $( seq 0 $(($variants-1)) )
do
    echo "Processing option $i"
    mkdir -p $bin_dir/$i $bin_dir/$i/insts $bin_dir/$i/states $bin_dir/$i/pos
    python3 preprocess.py --srcdict $data_dir/inst_dict.txt --only-source --trainpref $data_dir/$i/insts_train.txt --validpref $data_dir/$i/insts_valid.txt --destdir $bin_dir/$i/insts --dataset-impl mmap --workers 60
    python3 preprocess.py --srcdict $data_dir/state_dict.txt --only-source --trainpref $data_dir/$i/states_train.txt --validpref $data_dir/$i/states_valid.txt --destdir $bin_dir/$i/states --dataset-impl mmap --workers 60
	cp $data_dir/inst_dict.txt $bin_dir/inst_dict.txt
	cp $data_dir/state_dict.txt $bin_dir/state_dict.txt
done
for i in $( seq 0 $(($variants-1)) )
do
	python3 preprocess_pos.py moco $data_dir/$i/ $bin_dir/$i/pos/ &
done
wait
