#!/bin/sh

cache_dir=/scratch/scai/phd/aiz218326/zaki/matscibert/pretraining/.cache

# the commented part is the correct one :)
# train_file=/scratch/scai/phd/aiz218326/zaki/matscibert/pretraining/data_new/train_corpus.txt
# val_file=/scratch/scai/phd/aiz218326/zaki/matscibert/pretraining/data_new/val_corpus.txt
# train_norm_file=/scratch/scai/phd/aiz218326/zaki/matscibert/pretraining/data_new/train_corpus_norm.txt
# val_norm_file=/scratch/scai/phd/aiz218326/zaki/matscibert/pretraining/data_new/val_corpus_norm.txt
train_file=/scratch/civil/phd/cez198233/lms_trained/CemMatSciBERT/pretraining/data_cement/train_corpus.txt
val_file=/scratch/civil/phd/cez198233/lms_trained/CemMatSciBERT/pretraining/data_cement/val_corpus.txt
train_norm_file=/scratch/civil/phd/cez198233/lms_trained/CemMatSciBERT/pretraining/data_cement/train_corpus_norm.txt
val_norm_file=/scratch/civil/phd/cez198233/lms_trained/CemMatSciBERT/pretraining/data_cement/val_corpus_norm.txt
model_save_dir=/scratch/scai/phd/aiz218326/zaki/matscibert/pretraining/model



# python -u normalize_corpus.py --train_file $train_file --val_file $val_file --output_train_norm_file $train_norm_file --output_val_norm_file $val_norm_file
# python -u matscibert_pre_train.py --train_file $train_norm_file --val_file $val_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir

python -u matscibert_finetune_expt.py --train_file $train_norm_file --val_file $val_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir >out.txt 2>err.txt