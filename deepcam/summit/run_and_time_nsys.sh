#!/bin/bash
#BSUB -P csc330
#BSUB -W 02:00
##BSUB -w ended(######)
#BSUB -nnodes 8
#BSUB -alloc_flags "nvme smt4"
#BSUB -J DeepCam
#BSUB -o %J.out
#BSUB -e %J.err
# End LSF directives and begin shell commands

# Run parameters
export BATCHSIZE=2
export DO_PROFILING='false' # true or false

# Setup software environment
module purge
source activate mlperf_deepcam # TODO
module unload darshan          # TODO
module load pytorch/v1.6.0-gpu # TODO
module load cuda/10.2.89       # TODO
module load gcc                # TODO
module load mpich              # TODO
module load nccl/2.5.6         # TODO
if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi
# TODO
export BASEMAPDATA=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
export PROJ_LIB=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:/global/homes/m/mrowan/code/mlperf-logging
export NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)


# Copy data to burst buffer # TODO
jsrun -n$NODES -a1 -c42 -r1 cp $WORLDWORK/stf011/junqi/imagenet/train/* $WORLDWORK/stf011/junqi/imagenet/validation/* /mnt/bb/$USER
export DATADIR=/mnt/bb/$USER
export PYTHONPATH=$(pwd):$PYTHONPATH # TODO

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

if [ "$DO_PROFILING" == "true" ]
then
    cat ./utils/run_with_profiling.sh >> log.${LSB_JOBID}
else
    cat ./utils/run.sh >> log.${LSB_JOBID}
fi

mv log.${LSB_JOBID} $LOG_DIR/
mv %J.out $LOG_DIR/
mv %J.err $LOG_DIR/


###################
# Job configuration
rankspernode=8
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
run_tag="deepcam_cgpu_${SLURM_JOB_ID}_nodes${SLURM_NNODES}_batch${BATCH_SIZE}"

data_dir_prefix="/global/cscratch1/sd/mrowan/hpc_mlperf_nsys_scripts/deepcam/data/cam5_data/All-Hist_small_split_${SLURM_NNODES}"
TAG='W'

output_dir=$SCRATCH/deepcam/results/$run_tag

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

SCRATCH_DIR=$SCRATCH/hpc_mlperf_nsys_scripts/deepcam/data_nsys/${SLURM_NNODES}_node
PROF_DIR=outputs_nodes${SLURM_NNODES}_batch${BATCH_SIZE}
mkdir $PROF_DIR

# Run training
srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 80 / ${rankspernode} )) \
     --cpu_bind=cores \
     -x LD_LIBRARY_PATH \
     ./launch_profile.sh \
     src/deepCam/train_hdf5_ddp.py ${output_dir} ${data_dir_prefix} ${SCRATCH_DIR} ${PROF_DIR}

