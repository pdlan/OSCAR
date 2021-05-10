#!/bin/bash
input_dir=../data-raw/poj-clone-detection/ProgramData
output_dir=./out
moco_path=../data-bin/pretrain
max_len=511
process=24
mkdir -p $output_dir/source/train $output_dir/source/valid $output_dir/source/test \
    $output_dir/ir/train $output_dir/ir/valid $output_dir/ir/test \
    $output_dir/json/train $output_dir/json/valid $output_dir/json/test \
    $output_dir/concat_json/train $output_dir/concat_json/valid $output_dir/concat_json/test \
    $output_dir/rawtext/train $output_dir/rawtext/valid $output_dir/rawtext/test \
    $output_dir/result/train $output_dir/result/valid $output_dir/result/test
python 1_process_source_code.py $input_dir $output_dir/source 104
python 2_generate_makefile.py $output_dir/source/train $output_dir/ir/train $output_dir/Makefile.train
python 2_generate_makefile.py $output_dir/source/valid $output_dir/ir/valid $output_dir/Makefile.valid
python 2_generate_makefile.py $output_dir/source/test $output_dir/ir/test $output_dir/Makefile.test
make -j$process -k -f $output_dir/Makefile.train
make -j$process -k -f $output_dir/Makefile.valid
make -j$process -k -f $output_dir/Makefile.test
python 3_ir_to_json.py $output_dir/ir/train $output_dir/json/train/ 104 $process $max_len
python 3_ir_to_json.py $output_dir/ir/valid $output_dir/json/valid/ 104 $process $max_len
python 3_ir_to_json.py $output_dir/ir/test $output_dir/json/test/ 104 $process $max_len
python 4_concat_json.py $output_dir/ir/train/ $output_dir/json/train/ $output_dir/concat_json/train/ label.txt 104
python 4_concat_json.py $output_dir/ir/valid/ $output_dir/json/valid/ $output_dir/concat_json/valid/ label.txt 104
python 4_concat_json.py $output_dir/ir/test/ $output_dir/json/test/ $output_dir/concat_json/test/ label.txt 104
python 5_json_to_rawtext.py $output_dir/concat_json/train $output_dir/rawtext/train $moco_path $max_len
python 5_json_to_rawtext.py $output_dir/concat_json/valid $output_dir/rawtext/valid $moco_path $max_len
python 5_json_to_rawtext.py $output_dir/concat_json/test $output_dir/rawtext/test $moco_path $max_len
cp $output_dir/rawtext/train/* $output_dir/result/train
cp $output_dir/concat_json/train/label.txt $output_dir/result/train
cp $output_dir/rawtext/valid/* $output_dir/result/valid
cp $output_dir/concat_json/valid/label.txt $output_dir/result/valid
cp $output_dir/rawtext/test/* $output_dir/result/test
cp $output_dir/concat_json/test/label.txt $output_dir/result/test
cd ../model
./scripts/classification_preprocess.sh ../process-poj-classification-data/out/result ../data-bin/poj-classification $moco_path
