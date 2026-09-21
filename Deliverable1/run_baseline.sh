#!/bin/sh
#SBATCH --job-name=baseline
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#Activate python environment
source $STORE/mypython/bin/activate

cd $LUSTRE/HPCTools/LabAI
time python run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --train_file $LUSTRE/HPCTools/LabAI/train-v2.0.json \
  --validation_file $LUSTRE/HPCTools/LabAI/dev-v2.0.json \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./baseline_result