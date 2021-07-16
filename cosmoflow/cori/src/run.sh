#!/bin/bash
#SBATCH -C gpu -c 10
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 0:15:00
#SBATCH -A nstaff
#SBATCH -J train-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

#.scripts/setup_cgpu.sh
#export HOROVOD_TIMELINE=./timeline.json

#set -x
#srun -l -u python train.py -d --rank-gpu $@
module load cgpu
module load tensorflow/2.4.1-gpu
export PYTHONPATH=/usr/common/software/tensorflow/2.4.1-gpu/bin/python

srun -n 8 -N 1 python train.py \
     --rank-gpu \
     --amp \
     --stage-dir=/tmp \
     --n-train=65536 \ # 65536  131072 262144 524288
     --n-valid=8192  \  # 8192   16384  32768  65536
     > 1node.out
     #--tensorboard \
     
