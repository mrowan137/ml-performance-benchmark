#!/bin/bash
#SBATCH -C gpu -c 10
#SBATCH -N 8
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -A nstaff
#SBATCH -J train-cgpu-8
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
srun -n 64 -N 8 python train.py --resume -d --rank-gpu --amp --stage-dir=/tmp --n-train=524288 --n-valid=65536 --output-dir=results/cosmo-008 >> 8node.out
