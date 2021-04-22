##!/bin/bash
#BSUB -P csc330
#BSUB -W 00:15
##BSUB -w ended(######)
#BSUB -nnodes 1
#BSUB -alloc_flags "nvme smt4"
#BSUB -J CosmoFlow_profile
#BSUB -o %J.out
#BSUB -e %J.err
## End LSF directives and begin shell commands

#output_dir,BATCHSIZE,data_dir, prof_dir? update CONFIG files

# Run parameters
export BATCHSIZE=1
export DO_PROFILING='false' # true or false

# Setup software environment
source ~/.mlperf_deepcam_profile
module load cuda/10.2.89
#module unload python
conda activate open-ce-0.1-0 # has basemap
module unload darshan

# TODO
export PYTHONPATH=$PYTHONPATH:/ccs/home/mrowan/code/mlperf-logging # Add as part of directory?
export NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
export CONFIG=src/configs/cosmo_runs_gpu_$NODES.yaml

#XLA environment
#export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR
#export PYTHONPATH=$(pwd):$PYTHONPATH # TODO


if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi

# Copy data to burst buffer # TODO
export data_dir="/ccs/home/mrowan/scratch/cosmoUniverse_2019_05_4parE_tf"
#jsrun -n$NODES -a1 -c42 -r1 cp -rL ./data/cam5_data/All-Hist_small_split_${NODES}/ /mnt/bb/$USER
#export data_dir=/mnt/bb/$USER

if [ "$DO_PROFILING" == "true" ]
then
    jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
          --bind=proportional-packed:7 \
          -x LD_LIBRARY_PATH \
          stdbuf -o0 \
          ./utils/run_with_profiling.sh
else
    jsrun -r1 -a6 -c7 -g6 \
          --bind=proportional-packed:7 \
          -x LD_LIBRARY_PATH \
          stdbuf -o0 \
          ./utils/run.sh
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


