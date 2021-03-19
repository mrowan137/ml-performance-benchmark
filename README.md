# ml-performance-benchmark
Performance benchmarking for ML/AI workloads.

## ResNet
Imagenet data: download training and validation datasets from http://www.image-net.org/challenges/LSVRC/2012/downloads

Untar the data with:
```
tar xf ILSVRC2012_img_val.tar -C $IMAGENET_DATA_HOME/validation
tar xf ILSVRC2012_img_train.tar -C $IMAGENET_DATA_HOME/train
```

Data must be converted to `TFRecords` format; this can be done with the script https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
```
python imagenet_to_gcs.py \
  --raw_data_dir=$IMAGENET_DATA_HOME \
  --local_scratch_dir=$IMAGENET_DATA_HOME/tf_records \
  --nogcs_upload
```

Resnet directory based off: https://code.ornl.gov/olcf-analytics/summit/distributed-deep-learning-examples


## DeepCam

Requires to install `mlperf-logging` package (see https://github.com/mlcommons/logging).  May install with:
```
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```
then add to PYTHONPATH.

Data: symlinks to files on Cori

DeepCam directory based off: https://bitbucket.org/kibrahim/hpc_mlperf_nsys_scripts
