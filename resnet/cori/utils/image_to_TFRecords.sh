export IMAGENET_DATA_HOME=/global/cscratch1/sd/wbhimji/imagenet/raw_data/
export TFRECORD_DATA_HOME=/global/cscratch1/sd/mrowan/imagenet_data/

python ../official/resnet/imagenet_to_gcs.py \
       --raw_data_dir=$IMAGENET_DATA_HOME \
       --local_scratch_dir=$TFRECORD_DATA_HOME \
       --nogcs_upload