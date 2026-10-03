#!/usr/bin/env bash

#PBS -N matscibert_finetune_main_2
#PBS -P m3rg.spons
#PBS -m bea
#PBS -M aiy257591@scai.iitd.ac.in
#PBS -l select=1:ncpus=8:ngpus=8:mem=80gb
#PBS -l walltime=72:00:00
#PBS -q standard

# Submit with:  qsub matscibert_finetune_main_2.sh

PROJECT_DIR=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT
PRETRAIN_DIR=$PROJECT_DIR/pretraining
NGPUS=8

################# Environment #################
cd "$PROJECT_DIR"
source ~/.bashrc
module load anaconda/2025.06
conda activate /home/scai/msr/aiy257591/envs/matscibert2

# Fix for GLIBCXX_3.4.31 not found: use the conda env's libstdc++
# export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export PYTHONUNBUFFERED=1

echo "==============================="
echo "Job ID: $PBS_JOBID"
echo "Node: $(cat "$PBS_NODEFILE")"
python -c "import os, torch; print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES')); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
echo "==============================="

################# Paths #################
cd "$PRETRAIN_DIR"
mkdir -p output/matscibert2 error/matscibert2

cache_dir=$PRETRAIN_DIR/cache
train_file=$PRETRAIN_DIR/cemcat_data/train_corpus_norm.txt
val_file=$PRETRAIN_DIR/cemcat_data/val_corpus_norm.txt
model_save_dir=$PRETRAIN_DIR/model/matscibert2
gpu_log=$PRETRAIN_DIR/matscibert_finetune_main_2_gpu_usage.log

################# GPU monitoring #################
nvidia-smi --loop=10 > "$gpu_log" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null' EXIT
echo "GPU monitoring PID: $SMI_PID"

################# Training #################
echo "Starting finetuning on $NGPUS GPUs (DDP via torchrun)"

torchrun --standalone --nproc_per_node="$NGPUS" matscibert_finetune_main_2.py \
    --train_file "$train_file" \
    --val_file "$val_file" \
    --model_save_dir "$model_save_dir" \
    --cache_dir "$cache_dir" \
    > output/matscibert2/out.txt 2> error/matscibert2/err.txt
status=$?

echo "GPU log saved to $gpu_log"
echo "Finetuning finished with exit code $status"
exit $status