#!/usr/bin/env bash

#PBS -N updated_matscibert_finetune
#PBS -P m3rg.spons
#PBS -m bea
#PBS -M $USER@scai.iitd.ac.in
#PBS -l select=1:ncpus=8:ngpus=8:mem=80gb
#PBS -l walltime=72:00:00
#PBS -q standard
#################  Environment #################################
echo "==============================="
echo "Job ID: $PBS_JOBID"
echo "Node:"
cat $PBS_NODEFILE
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import os, torch; print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES')); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
echo "==============================="

# changes current directory to the project root
cd /home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT

# important step to execute the .bashrc file
source ~/.bashrc

# load module
module load anaconda/2025.06

# environment directory
conda activate /home/scai/msr/aiy257591/envs/matscibert_new


# --- Main Execution  ---
echo "Starting finetuning of MatSciBERT using the script 'updated_matscibert_finetune_expt.py'"

# changing directory to pretraining folder
cd pretraining/

echo "Starting GPU monitoring in background..."
nvidia-smi --loop=10 > updated_matscibert_finetune_gpu_usage.log &
SMI_PID=$!

# Always stop the GPU monitoring process when this script exits
trap 'kill "$SMI_PID" 2>/dev/null' EXIT
echo "GPU monitoring PID: $SMI_PID"

echo "Running the file 'updated_matscibert_finetune_expt.py'"



# HOW TO RUN:
# 1) Normal full training -----> bash run_finetuning.sh
# 2) Debug / test run -----> DEBUG=1 bash run_finetuning.sh

# Fix for GLIBCXX_3.4.31 not found — use conda env's libstdc++ instead of system one
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

# train-val directories
cache_dir=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cache
train_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/train_corpus.txt
val_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/val_corpus.txt
train_norm_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/train_corpus_norm.txt
val_norm_file=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/cemcat_data/val_corpus_norm.txt
model_save_dir=/home/scai/msr/aiy257591/flash/Projects/MatSci/MatSciBERT/pretraining/model
model_save_dir_debug="${model_save_dir}_debug"


# for debug mode uncomment the below part:
# echo "Running in DEBUG mode"
# python -u updated_matscibert_finetune_expt.py --train_file $train_norm_file --val_file $val_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir --debug $model_save_dir_debug >output/out.txt 2>error/err.txt

python -u updated_matscibert_finetune_expt.py --train_file $train_norm_file --val_file $val_norm_file --model_save_dir $model_save_dir --cache_dir $cache_dir >output/out.txt 2>error/err.txt

# Stop GPU Monitoring
kill $SMI_PID
echo "GPU logs saved to Q3/gpu_usage_Q3.log"

echo "Finetuning Completed"