#!/bin/bash
processes=24
variants=20
rawdata_path=../data-raw/pretrain
max_len=511
rate="1 1 1 1 1 1 1 0.3 1 0.2 1"
programs="apache imagemagick linux-module linux-vmlinux mplayer openblas postgresql tensorflow gcc blender firefox"

make
mkdir -p out
ulimit -n 1048576
for program in $programs
do
    echo Processing $program
    mkdir -p out/1/$program/
    time python3 1_ir2json.py $rawdata_path/$program out/1/$program/ 1.0 $variants $processes
done

in_dirs=()
for program in $programs
do
    in_dirs+=(./out/1/$program)
done
tmp_dir=/dev/shm/`uuidgen`
out_dir=out/2/
mkdir -p $tmp_dir $out_dir
./2_concatjson $out_dir $tmp_dir $out_dir/state_dict.txt $variants $processes ${in_dirs[@]} $rate
rm -rf $tmp_dir

mkdir -p out/3
python3 3_processir.py out/2 out/3 $variants $processes ../bin/irlexer ../bin/fast $max_len

mkdir -p out/4
python3 4_json2rawtext.py out/2 out/3 out/4 $variants $processes out/2/state_dict.txt 50

./5_buildvocab out/4 $variants out/4/inst_dict.txt out/4/state_dict.txt
mkdir -p ../data-bin/pretrain
cp out/3/bpe_codes ../data-bin/pretrain
cd ../model
./scripts/moco_preprocess.sh ../process-pretrain-data/out/4/ ../data-bin/pretrain $variants
