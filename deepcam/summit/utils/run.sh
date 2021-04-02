#!/bin/bash

# Where to store results and logfiles
run_tag="${LSB_JOBID}"
output_dir="results/${NODES}_nodes_batchsize_${BATCHSIZE}_j${run_tag}"

#if [ $PMIX_RANK -eq 0 ]
#then
mkdir -p ${output_dir}
touch ${output_dir}/train.out
#fi

python src/deepCam/train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --max_epochs 70 \
     --amp_opt_level O1 \
     --local_batch_size $BATCHSIZE |& tee -a ${output_dir}/train.out
