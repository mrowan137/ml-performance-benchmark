#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 5
#DW persistentdw name=resnetbb
#DW stage_in source=/global/cscratch1/sd/mrowan/scratch/imagenet destination=$DW_PERSISTENT_STRIPED_resnetbb/imagenet type=directory

mkdir $DW_PERSISTENT_STRIPED_resnetbb/imagenet/all_data
mv $DW_PERSISTENT_STRIPED_resnetbb/imagenet/train/* $DW_PERSISTENT_STRIPED_resnetbb/imagenet/all_data
mv $DW_PERSISTENT_STRIPED_resnetbb/imagenet/validation/* $DW_PERSISTENT_STRIPED_resnetbb/imagenet/all_data

echo "Data successfully staged into $DW_PERSISTENT_STRIPED_resnetbb"
ls $DW_PERSISTENT_STRIPED_resnetbb
