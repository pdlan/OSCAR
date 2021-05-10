#!/bin/bash
in_dir=$1
in_file=$2
out_dir=$3
max_len=$4
mkdir -p $out_dir $out_dir/stripped $out_dir/ir $out_dir/json $out_dir/eval_json $out_dir/result
variants=O0-gcc7.5.0-amd64,O1-gcc7.5.0-amd64,O2-gcc7.5.0-amd64,O3-gcc7.5.0-amd64 #,O0-gcc7.5.0-aarch64,O1-gcc7.5.0-aarch64,O2-gcc7.5.0-aarch64,O3-gcc7.5.0-aarch64
python 1_generate_ground_truth.py $in_dir $in_file $variants $out_dir/function_info.json
python 2_rename_symbols.py $in_dir $in_file $variants $out_dir/stripped
python 3_bin2llvm.py $out_dir/stripped $in_file $variants $out_dir/ir
python 4_llvm2json.py $out_dir/ir $in_file $variants $out_dir/json $max_len
python 5_2_generate_eval_data.py $out_dir/json $in_file $variants $out_dir/eval_json $out_dir/function_info.json $out_dir/ir $out_dir/stripped $out_dir/matched_functions.json
python 6_json2rawtext.py $out_dir/eval_json $in_file $variants $out_dir/result ../data-bin/pretrain/inst_dict.txt ../data-bin/pretrain/state_dict.txt $max_len
