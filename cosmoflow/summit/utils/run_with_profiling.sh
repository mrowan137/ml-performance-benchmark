#!/bin/bash

# Where to store results and logfiles
run_tag="${LSB_JOBID}"
export output_dir="results/${NODES}_nodes_batchsize_${batch_size}_j${run_tag}"
profile_dir="${output_dir}/profiling_results"

#if [ $PMIX_RANK -eq 0 ]
#then
mkdir -p ${output_dir}
mkdir -p ${profile_dir}
touch ${output_dir}/train.out
#fi
prof_file=${profile_dir}/nsys.${LSB_JOBID}.r${PMIX_RANK}.w${LSB_JOB_NUMPROC}

nsys profile -o $prof_file -t cuda \
     python src/train.py -d \
     --batch-size=$batch_size \
     --data-dir=$data_dir \
     --output-dir=$output_dir \
     --rank-gpu $CONFIG |& tee -a ${output_dir}/train.out
