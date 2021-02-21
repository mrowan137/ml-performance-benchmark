#!/bin/bash
#BSUB -P csc330
#BSUB -W 04:00
#BSUB -nnodes 16
#BSUB -alloc_flags "nvme smt4"
#BSUB -J ResNet50
#BSUB -o %J.out
#BSUB -e %J.err
# End LSF directives and begin shell commands

# Environment setup
export NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
export BATCHSIZE=128
export STRATEGY='horovod'  # horovod or multi_worker_mirrored
export DATA_MODE='real'    # real or synthetic
export DO_PROFILING='false' # true or false
#export NCCL_DEBUG_SUBSYS=COLL

source $WORLDWORK/stf011/junqi/native-build/latest/1.14.0/env.sh
if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi

if [ "$DATA_MODE" == "real" ]
then
    #copy imagenet data to SSD 
    jsrun -n$NODES -a1 -c42 -r1 cp $WORLDWORK/csc330/mrowan/imagenet/train/* $WORLDWORK/csc330/mrowan/imagenet/validation/* /mnt/bb/$USER
    
fi

rm -rf $MODELDIR
export MODELDIR=/mnt/bb/$USER/models/model_dir_${NODES}_nodes
export DATADIR=/mnt/bb/$USER

#XLA environment
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH

if [ "$STRATEGY" == "horovod" ]
then
    if [ "$DO_PROFILING" == "true" ]
    then
    jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
          --bind=proportional-packed:7 \
          --launch_distribution=packed \
          stdbuf -o0 \
          ./utils/launch.sh "./utils/run_with_profiling.sh"
    else
        jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
              --bind=proportional-packed:7 \
              --launch_distribution=packed \
              stdbuf -o0 \
              ./utils/launch.sh "./utils/run.sh"
    fi
else
    if [ "$DO_PROFILING" == "true" ]
    then
        jsrun -n${NODES} -a1 -c42 -g6 -r1 -b none stdbuf -o0 "./utils/run_with_profiling.sh"
    else
        jsrun -n${NODES} -a1 -c42 -g6 -r1 -b none stdbuf -o0 "./utils/run.sh"
    fi
fi

if [ "$DO_PROFILING" == "true" ]
then
    cat ./utils/run_with_profiling.sh >> log.${LSB_JOBID}
else
    cat ./utils/run.sh >> log.${LSB_JOBID}
fi

mv log.${LSB_JOBID} $LOG_DIR/
mv %J.out $LOG_DIR/
mv %J.err $LOG_DIR/
