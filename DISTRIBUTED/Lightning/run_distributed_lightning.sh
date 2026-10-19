#!/bin/sh
#SBATCH --job-name=distributed__lightning_2gpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=02:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../error/%x_%j.err

#Activate python environment
source $STORE/mypython/bin/activate

#Install needed packages for the run_qa.py script (datasets module for SQUAD)
pip install -q -r requirements.txt

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > gpu_usage_light.csv &

srun python -u run_qa_lightning.py \
  --model_name_or_path google-bert/bert-base-uncased \
  --dataset_name rajpurkar/squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $STORE/hpctools/results_lightning_$SLURM_JOB_ID