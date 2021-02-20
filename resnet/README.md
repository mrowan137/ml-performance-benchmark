# Running TensorFlow official models in `MultiworkerMirroredStrategy` or `Horovod` mode on Summit

The majority of the code is from TensorFlow's [official implementation of ResNet-50](https://github.com/tensorflow/models/tree/master/official/resnet). A minimal amount of modification is made to the `imagenet_main.py` to scale up the code on Summit. While this strategy is slightly slowered than Horovod, the setup is easier and the modification you need to make is only at the beginning.

## Requirement

You need to have access to `/gpfs/alpine/world-shared` directory on Summit.

## How to run

1. Navigate yourself to this folder

2. Type `bsub job.lsf` to submit the job

The job script runs the official ResNet-50 implementation using `MultiworkerMirroredStrategy`. Notice that the only modification on `imagenet_main.py` is the intial configuration setup.

```python
import os
import subprocess
import json

# Get a list of compute nodes allocated for your job
get_cnodes = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login)".format(os.environ['LSB_DJOB_HOSTFILE'])
cnodes = subprocess.check_output(get_cnodes, shell=True)
cnodes = str(cnodes)[2:-3].split(' ')
nodes_list = [c + ":2222" for c in cnodes] # Add a port number

# Get the rank of the compute node that is running on
index = int(os.environ['PMIX_RANK'])

# Set the TF_CONFIG environment variable to configure the cluster setting. 
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': nodes_list
    },
    'task': {'type': 'worker', 'index': index} 
})
```

