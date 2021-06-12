#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH -t 4:00:00
#SBATCH -A nstaff
#SBATCH --exclusive
#SBATCH -J resnet50-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# End Slurm directives and begin shell commands

# Run parameters
export BATCHSIZE=64
export STRATEGY='horovod'    # horovod or multi_worker_mirrored
export DATA_MODE='real' # real or synthetic
export DO_PROFILING='false'  # true or false
export DO_NCCL_DEBUG='false'  # true or false

# Setup software environment
module purge
module load cgpu

module load tensorflow/gpu-1.15.0-py37
export PYTHONPATH=/global/cscratch1/sd/mrowan/ml-performance-benchmark/resnet/cori:/usr/common/software/tensorflow/gpu-tensorflow/1.15.0-py37/bin/python

# module load tensorflow/gpu-2.2.0-py37
# export PYTHONPATH=/global/cscratch1/sd/mrowan/ml-performance-benchmark/resnet/cori:/usr/common/software/tensorflow/gpu-tensorflow/2.2.0-py37/bin/python
# module load tensorflow/2.4.1-gpu
# export PYTHONPATH=/global/cscratch1/sd/mrowan/ml-performance-benchmark/resnet/cori:/usr/common/software/tensorflow/2.4.1-gpu/bin/python

# export MODELDIR=/mnt/bb/$USER/models/model_dir_${NODES}_nodes
export NODES=$SLURM_NNODES
# #rm -rf $MODELDIR

# # XLA environment
# source $WORLDWORK/stf011/junqi/native-build/latest/1.14.0/env.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
# export PYTHONPATH=$(pwd):$PYTHONPATH

if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi

if [ "$DATA_MODE" == "real" ]
then
    #copy imagenet data to BB
    #./utils/stage_data_bb.sh
    #srun -n 1 -c $((NODES*8*10)) cp /global/cscratch1/sd/mrowan/scratch/imagenet/train/* /global/cscratch1/sd/mrowan/scratch/imagenet/validation/* /mnt/bb/$USER
fi
#export DATADIR=/mnt/bb/$USER

#DW persistentdw name=cosmobb
export DATADIR=/global/cscratch1/sd/mrowan/imagenet/all_data

if [ "$STRATEGY" == "horovod" ]
then
    if [ "$DO_PROFILING" == "true" ]
    then
        echo "then"
        srun -N $NODES -n $((NODES*8)) -c 10 \
             --cpu-bind=cores \
             --mem=30GB \
             ./utils/run_with_profiling.sh
    else
        echo "else"
        srun -N $NODES -n $((NODES*8)) -c 10 \
             --cpu-bind=cores \
             --mem=30GB \
             ./utils/run.sh
    fi
else
    if [ "$DO_PROFILING" == "true" ]
    then
        jsrun -n${NODES} -a1 -c42 -g6 -r1 -b none stdbuf -o0 "./utils/run_with_profiling.sh"
    else
        jsrun -n${NODES} -a1 -c42 -g6 -r1 -b none stdbuf -o0 "./utils/run.sh"
    fi
fi

# if [ "$DO_PROFILING" == "true" ]
# then
#     cat ./utils/run_with_profiling.sh >> log.${LSB_JOBID}
# else
#     cat ./utils/run.sh >> log.${LSB_JOBID}
# fi

# mv log.${LSB_JOBID} $LOG_DIR/
# mv %J.out $LOG_DIR/
# mv %J.err $LOG_DIR/
