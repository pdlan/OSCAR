#!/bin/bash
max_len=511
./process_train.sh ../data-raw/bindiff/gnutls-3.5.19/ libgnutls.so ./data/gnutls $max_len &
./process_train.sh ../data-raw/bindiff/wolfssl-4.6.0/ libwolfssl.so ./data/wolfssl $max_len &
./process_train.sh ../data-raw/bindiff/openssh-sshd-8.4/ sshd ./data/openssh $max_len &
./process_train.sh ../data-raw/bindiff/bash-5.1/ bash ./data/bash $max_len &
./process_train.sh ../data-raw/bindiff/bzip2-1.0.8/ bzip2 ./data/bzip2 $max_len &
./process_train.sh ../data-raw/bindiff/p7zip-16.02/ 7za ./data/p7zip $max_len &
./process_train.sh ../data-raw/bindiff/cairo-1.16.0/ libcairo.so ./data/cairo $max_len &
./process_train.sh ../data-raw/bindiff/zsh-5.8/ zsh ./data/zsh $max_len &
./process_train.sh ../data-raw/bindiff/x265-3.4/ libx265.so ./data/x265/ $max_len &
./process_train.sh ../data-raw/bindiff/emacs-27.1/ emacs ./data/emacs $max_len &
./process_train.sh ../data-raw/bindiff/mariadb-10.5.8/ mariadb ./data/mariadb $max_len &
./process_train.sh ../data-raw/bindiff/apache-2.4.46/ httpd ./data/apache $max_len &
./process_train.sh ../data-raw/bindiff/nginx-1.18.0/ nginx ./data/nginx $max_len &
./process_train.sh ../data-raw/bindiff/git-2.30.0/ git ./data/git $max_len &
./process_train.sh ../data-raw/bindiff/vim-8.1/ vim ./data/vim $max_len &
./process_eval.sh ../data-raw/bindiff/openssl-libcrypto-1.1.1i/ libcrypto.so ./data/openssl $max_len &
./process_eval.sh ../data-raw/bindiff/busybox-1.28.0/ busybox_unstripped ./data/busybox $max_len &
./process_eval.sh ../data-raw/bindiff/sqlite-3.34.0/ sqlite3 ./data/sqlite $max_len &
./process_eval.sh ../data-raw/bindiff/zlib-1.2.11/ libz.so ./data/zlib $max_len &
./process_eval.sh ../data-raw/bindiff/imagemagick-7.0.10/ libMagickCore.so ./data/imagemagick $max_len &
./process_eval.sh ../data-raw/bindiff/libcurl-7.74.0/ libcurl.so ./data/libcurl $max_len &
./process_eval.sh ../data-raw/bindiff/libtomcrypt-1.18.2/ libtomcrypt.so ./data/libtomcrypt $max_len &
wait
variants=O0-gcc7.5.0-amd64,O1-gcc7.5.0-amd64,O2-gcc7.5.0-amd64,O3-gcc7.5.0-amd64
mkdir -p train valid
python3 7_concat.py train $variants ./data/gnutls/result ./data/wolfssl/result ./data/openssh/result \
./data/bash/result ./data/bzip2/result ./data/p7zip/result ./data/zsh/result ./data/emacs/result ./data/mariadb/result \
./data/apache/result ./data/nginx/result ./data/git/result ./data/vim/result

python3 7_concat.py valid $variants ./data/cairo/result ./data/x265/result
mkdir -p ../data-bin/bindiff ../data-bin/bindiff-eval
cd ../model
./scripts/bindiff_train_preprocess.sh ../process-bindiff-data/ ../data-bin/bindiff ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/busybox/result ../data-bin/bindiff-eval/busybox ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/openssl/result ../data-bin/bindiff-eval/openssl ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/sqlite/result ../data-bin/bindiff-eval/sqlite ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/zlib/result ../data-bin/bindiff-eval/zlib ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/imagemagick/result ../data-bin/bindiff-eval/imagemagick ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/libcurl/result ../data-bin/bindiff-eval/libcurl ../data-bin/pretrain
./scripts/bindiff_eval_preprocess.sh ../process-bindiff-data/data/libtomcrypt/result ../data-bin/bindiff-eval/libtomcrypt ../data-bin/pretrain
cp ../process-bindiff-data/data/busybox/*.json ../data-bin/bindiff-eval/busybox/
cp ../process-bindiff-data/data/openssl/*.json ../data-bin/bindiff-eval/openssl/
cp ../process-bindiff-data/data/sqlite/*.json ../data-bin/bindiff-eval/sqlite/
cp ../process-bindiff-data/data/zlib/*.json ../data-bin/bindiff-eval/zlib/
cp ../process-bindiff-data/data/imagemagick/*.json ../data-bin/bindiff-eval/imagemagick/
cp ../process-bindiff-data/data/libcurl/*.json ../data-bin/bindiff-eval/libcurl/
cp ../process-bindiff-data/data/libtomcrypt/*.json ../data-bin/bindiff-eval/libtomcrypt/
