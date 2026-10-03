#!/bin/sh

# HOW TO RUN:
# 1) Normal full training
# bash run_finetuning.sh

# 2) Debug / test run
# DEBUG=1 bash run_finetuning.sh

# Fix for GLIBCXX_3.4.31 not found — use conda env's libstdc++ instead of system one
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

cache_dir=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cache

train_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/train_corpus.txt
val_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/val_corpus.txt
train_norm_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/train_corpus_norm.txt
val_norm_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/val_corpus_norm.txt
model_save_dir=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/model

# Set DEBUG=1 to run on small subset: DEBUG=1 bash run_finetuning.sh
DEBUG_FLAG=""
if [ "${DEBUG:-0}" = "1" ]; then
    DEBUG_FLAG="--debug"
    model_save_dir="${model_save_dir}_debug"
    echo "Running in DEBUG mode"
fi

python -u updated_matscibert_finetune_expt.py --train_file $train_norm_file --val_file $val_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir $DEBUG_FLAG >out1.txt 2>err1.txt