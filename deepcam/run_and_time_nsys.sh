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
BATCH_SIZE=2

# Setup software environment
#module unload darshan
#module load cgpu
#module load pytorch/v1.6.0-gpu
#module load gcc/7.3.0
#module load mpich/3.3.1-debug
#module load cuda/11.1.1
source activate mlperf_deepcam
#module unload darshan
#module load cgpu
#module load pytorch/v1.6.0-gpu
#module load nsight-systems
module unload darshan
module load cgpu
module load pytorch/v1.6.0-gpu
module load cuda/10.2.89
module load gcc
module load mpich
#module load nccl
module load nccl/2.5.6
#module load pmi
module load nsight-systems
#module load nsight-systems/2020.5.1
export BASEMAPDATA=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
export PROJ_LIB=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
export PYTHONPATH=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:/global/homes/m/mrowan/code/mlperf-logging
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/common/software/mpich/3.3.1-debug/gcc/8.3.0/lib:/usr/common/software/sles15_cgpu/gcc/8.3.0/lib64:/usr/common/software/nccl/2.5.6/cuda-10.2.89/lib:/usr/common/software/sles15_cgpu/cuda/10.2.89/extras/CUPTI/lib64:/usr/common/software/sles15_cgpu/cuda/10.2.89/lib64:/global/common/cori_cle7/software/jdk/1.8.0_202/lib:/opt/esslurm/lib64:/usr/common/software/nccl/2.5.6/cuda-11.1.1/lib:/opt/cray/job/2.2.4-7.0.1.1_3.43__g36b56f4.ari/lib64:/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64:/usr/lib64:/usr/common/software/sles15_cgpu/cuda/10.2.89/lib64/stubs
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=eth0,eth3,eth0_144,eth0_224
#export NCCL_SOCKET_IFNAME=eth0,eth3,eth0_144,eth0_224,ib0,ib1,ib2,ib3,ib4,ib5,ib6

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

