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
#SBATCH --exclusive

#Activate python environment
source $STORE/mypython/bin/activate

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#Install needed packages for the run_qa.py script (datasets module for SQUAD)
pip install -q -r requirements.txt

perf stat python run_qa.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name rajpurkar/squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/