#!/bin/bash

# Where to store results and logfiles
run_tag="${LSB_JOBID}"
output_dir="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${run_tag}"
profile_dir="${output_dir}/profiling_results"

#if [ $PMIX_RANK -eq 0 ]
#then
mkdir -p ${output_dir}
mkdir -p ${profile_dir}
touch ${output_dir}/train.out
#fi
prof_file=${profile_dir}/nsys.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}

nsys profile -o $prof_file -t cuda \
python src/deepCam/train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --max_epochs 3 \
     --amp_opt_level O1 \
     --local_batch_size $BATCHSIZE |& tee -a ${output_dir}/train.out
