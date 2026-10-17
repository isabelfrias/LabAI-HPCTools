#!/bin/sh
#SBATCH --job-name=distributed_acc_2GPU
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=error/%x_%j.err

#Activate python environment
source $STORE/mypython/bin/activate

#Install needed packages for the run_qa.py script (datasets module for SQUAD)
pip install -q -r requirements.txt

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 5 > gpu_usage_acc.csv &

export HF_HOME=$STORE/hf_cache
export PYTHONUNBUFFERED=1

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name rajpurkar/squad \
  --do_train \
  --do_eval \
  --bf16 \
  --per_device_train_batch_size 16 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $STORE/hpctools/results_acc_2gpus_$SLURM_JOB_ID