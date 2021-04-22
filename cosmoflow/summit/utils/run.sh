#!/bin/bash

# Where to store results and logfiles
run_tag="${LSB_JOBID}"
export output_dir="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${run_tag}"

#if [ $PMIX_RANK -eq 0 ]
#then
mkdir -p ${output_dir}
touch ${output_dir}/train.out
#fi

python src/train.py -d --rank-gpu $CONFIG |& tee -a ${output_dir}/train.out
