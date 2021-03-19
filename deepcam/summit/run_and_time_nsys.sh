##!/bin/bash
#BSUB -P csc330
#BSUB -W 01:00
##BSUB -w ended(######)
#BSUB -nnodes 1
#BSUB -alloc_flags "nvme smt4"
#BSUB -J DeepCam_bs2
#BSUB -o %J.out
#BSUB -e %J.err
## End LSF directives and begin shell commands

# Run parameters
export BATCHSIZE=8
export DO_PROFILING='false' # true or false

# Setup software environment
source ~/.bashrc
module unload python
module load open-ce
#source activate mlperf_deepcam # TODO
module unload darshan          # TODO
#module load open-ce
#module load pytorch/v1.6.0-gpu # TODO
#module load cuda/10.2.89       # TODO
#module load gcc                # TODO
#module load mpich              # TODO
#module load nccl/2.5.6         # TODO

if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi
# TODO
#export BASEMAPDATA=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
#export PROJ_LIB=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
#export PYTHONPATH=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:/global/homes/m/mrowan/code/mlperf-logging
export PYTHONPATH=$PYTHONPATH:/ccs/home/mrowan/code/mlperf-logging
export NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)


# Copy data to burst buffer # TODO
#jsrun -n$NODES -a1 -c42 -r1 cp $WORLDWORK/stf011/junqi/imagenet/train/* $WORLDWORK/stf011/junqi/imagenet/validation/* /mnt/bb/$USER
#export DATADIR=/mnt/bb/$USER
export DATADIR="/ccs/home/mrowan/scratch/ml-performance-benchmark/deepcam/summit/data/cam5_data/All-Hist_small_split_${NODES}"

#source $WORLDWORK/stf011/junqi/native-build/latest/1.14.0/env.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH # TODO

rankspernode=6
totalranks=$(( ${NODES} * ${rankspernode} ))
run_tag="deepcam_${LSB_JOBID}_nodes${NODES}_batch${BATCHSIZE}"

TAG='W'

output_dir=/ccs/home/mrowan/scratch/ml-performance-benchmark/deepcam/summit/results/$run_tag

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

SCRATCH_DIR=/ccs/home/mrowan/scratch/ml-performance-benchmark/deepcam/summit/data_nsys/${NODES}_node
PROF_DIR=outputs_nodes${NODES}_batch${BATCHSIZE}
mkdir $PROF_DIR

if [ "$DO_PROFILING" == "true" ]
then
    jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
          --bind=proportional-packed:7 \
          --launch_distribution=packed \
          stdbuf -o0 \
          ./utils/launch.sh "./utils/run_with_profiling.sh"
else
    # jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
    #       --bind=proportional-packed:7 \
    #       --launch_distribution=packed \
    #       stdbuf -o0 \
    #       ./utils/launch.sh "./utils/run.sh"
    
    # Run training
    jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
          --bind=proportional-packed:7 \
          -x LD_LIBRARY_PATH \
          stdbuf -o0 \
          ./launch_profile.sh \
          src/deepCam/train_hdf5_ddp.py ${output_dir} ${DATADIR} ${SCRATCH_DIR} ${PROF_DIR}
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


