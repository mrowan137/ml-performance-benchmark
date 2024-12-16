# ml-performance-benchmark
Performance benchmarking of ML/AI workloads ResNet, DeepCam, and CosmoFlow; work in support of [Architectural Requirements for Deep Learning Workloads in HPC Environments (PMBS21)](https://ieeexplore.ieee.org/document/9652793).


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

Resnet directory: https://code.ornl.gov/olcf-analytics/summit/distributed-deep-learning-examples


## DeepCam

DeepCam directory: https://github.com/sparticlesteve/mlperf-deepcam/tree/nersc-dev

## CosmoFlow
    
CosmoFlow directory: https://github.com/sparticlesteve/cosmoflow-benchmark
