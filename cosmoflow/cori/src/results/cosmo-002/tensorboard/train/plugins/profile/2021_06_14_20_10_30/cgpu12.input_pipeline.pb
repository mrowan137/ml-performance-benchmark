$	����SZ@�����?�Hh˹Z@!�۟��uZ@$	�����]�?��OX�?=EJ-�O�?!�'L/=�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�Hh˹Z@o��}Un�?1�)��X@A������?I�$��@Y2��4�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�۟��uZ@y!�x�?1��o$X@A0��\��?I��4}F@Y��E��\�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�i���tZ@?e� @1��(&o9X@Av5y�j�?I��c��?Y� v��y�?*��"��*A�ʡE�x&A)      �=2�
VIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2	X歺@!|�	m�x@@)X歺@1|�	m�x@@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2�CD,�@!�l��F@)`L��f@1��	��&@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[24]::TFRecord|~�d@!�e.��#@)|~�d@1�e.��#@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[23]::TFRecordM��f�X_@!��3�'A@)M��f�X_@1��3�'A@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[29]::TFRecord!%̴H\@!�M�o3@)!%̴H\@1�M�o3@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[26]::TFRecordp��s��Y@!����:�@)p��s��Y@1����:�@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[21]::TFRecordT㥛�Y@!��K��@)T㥛�Y@1��K��@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[28]::TFRecord��m�Q@!�m=(�@)��m�Q@1�m=(�@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[25]::TFRecord�8�ߡ Q@!�.�z�@)�8�ߡ Q@1�.�z�@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[22]::TFRecordU���NP@!���= @)U���NP@1���= @:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[27]::TFRecord7o��O@!:�����@)7o��O@1:�����@:Advanced file read2�
_Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap	ճ ����@!��[���K@)�GĔ�:@1�qn���?:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[20]::TFRecord�R	O���?!y*F`ߍ?)�R	O���?1y*F`ߍ?:Advanced file read2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat	��͋�@!Bz�y@@)�C�M�?1<r���@�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismr6�,�?!���0r?)r6�,�?1�=a�.f?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�l:�Y�?!<��eD\?)�l:�Y�?1<��eD\?:Preprocessing2�
hIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap::Shuffle
~��!ƛ?!�~Z�>�[?)~��!ƛ?1�~Z�>�[?:Preprocessing2F
Iterator::Model5z5@i��?!�&���u?)��i܋?1�Gp&v�K?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��7)xq�?I�Iۺ�# @Q���,U�V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	����q	@�b_n�@y!�x�?!?e� @	!       "$	k�O�>'X@Z(�NC��?�)��X@!��(&o9X@*	!       2$	a�Lt���?$�{�^q�?������?!0��\��?:$	b���L�@%�� K}@��c��?!��4}F@B	!       J$	0s�a��?��iu׿?2��4�?!��E��\�?R	!       Z$	0s�a��?��iu׿?2��4�?!��E��\�?b	!       JGPUY��7)xq�?b q�Iۺ�# @y���,U�V@