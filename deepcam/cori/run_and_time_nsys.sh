#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH	--nodes=2
#SBATCH -C gpu
#SBATCH -A nstaff
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
##SBATCH --time-min 2:00:00
#SBATCH --time 4:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Run parameters
export BATCH_SIZE=2

# Setup software environment
source activate mlperf_deepcam
module unload darshan
module load cgpu
module load pytorch/v1.6.0-gpu
module load cuda/10.2.89
module load gcc
module load mpich
module load nccl/2.5.6
module load nsight-systems
export BASEMAPDATA=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
export PROJ_LIB=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:/global/homes/m/mrowan/code/mlperf-logging

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

