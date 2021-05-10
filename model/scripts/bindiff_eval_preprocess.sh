#!/bin/bash
data_dir=$1
bin_dir=$2
moco_path=$3
variants="O0-gcc7.5.0-amd64 O1-gcc7.5.0-amd64 O2-gcc7.5.0-amd64 O3-gcc7.5.0-amd64"
#O0-gcc7.5.0-aarch64 O1-gcc7.5.0-aarch64 O2-gcc7.5.0-aarch64 O3-gcc7.5.0-aarch64"
for v in $variants
do
	mkdir -p $bin_dir/$v $bin_dir/$v/states $bin_dir/$v/states $bin_dir/$v/pos $bin_dir/$v/label $bin_dir/$v/test/label
	python3 preprocess.py --srcdict $moco_path/inst_dict.txt --only-source --testpref $data_dir/$v/insts.txt --destdir $bin_dir/$v/insts --dataset-impl mmap --workers 60
	python3 preprocess.py --srcdict $moco_path/state_dict.txt --only-source --testpref $data_dir/$v/states.txt --destdir $bin_dir/$v/states --dataset-impl mmap --workers 60
	python3 preprocess_pos.py bindiff $data_dir/$v/pos.json $bin_dir/$v/pos/
	cp $moco_path/inst_dict.txt $bin_dir/$v/
	cp $moco_path/state_dict.txt $bin_dir/$v
done
