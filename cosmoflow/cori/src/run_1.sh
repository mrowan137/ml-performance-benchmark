#!/bin/bash
#SBATCH -C gpu -c 10
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -A nstaff
#SBATCH -J train-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

#.scripts/setup_cgpu.sh
#export HOROVOD_TIMELINE=./timeline.json

#set -x
#srun -l -u python train.py -d --rank-gpu $@
module purge
module load cgpu
module load tensorflow/2.4.1-gpu
export PYTHONPATH=/usr/common/software/tensorflow/2.4.1-gpu/bin/python

# 65536  131072 262144 524288
# 8192   16384  32768  65536
#srun -n 8 -N 1 python train.py -d --rank-gpu --amp --stage-dir=/tmp --n-train=65536 --n-valid=8192 > 1node.out
#srun -n 8 -N 1 python train.py -d --rank-gpu --stage-dir=/tmp --n-train=65536 --n-valid=8192 >> 1node_noamp.out
#srun -n 8 -N 1 python train.py -d --rank-gpu --stage-dir=/tmp --n-train=32768 --n-valid=4096 >> 1node_noamp.out
#srun -n 8 -N 1 python train.py -d --amp --rank-gpu --n-train=24576 --n-valid=3072 --stage-dir=/tmp >> 1node_yesamp.out
srun -n 8 -N 1 python train.py -d       --rank-gpu --n-train=24576 --n-valid=3072 --stage-dir=/tmp >> 1node_noamp.out
     
