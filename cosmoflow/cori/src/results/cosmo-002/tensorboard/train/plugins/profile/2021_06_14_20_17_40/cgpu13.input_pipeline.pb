$	�{Q��Y@b�;���?�s��йY@!���N�Z@	TZy=��?xꂆ�a�?![���s��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8���N�Z@i������?1L�g�X@A�XİØ�?I�f�v�@Yl_@/ܹ�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8��$\�Y@����?1��ډ�X@A���9�@I���?Y��ZӼ��?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/�s��йY@���t @1�����W@A��ۂ���?I�%��@*e;�O�[A/�$�"#A2�
VIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2;]���c@!�ڻi�5@);]���c@1�ڻi�5@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2�nJy�W@!�0��g=)@)���W@1�Pk3�9)@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[28]::TFRecord0F$
-dU@!��U3>�&@)0F$
-dU@1��U3>�&@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[29]::TFRecordC��!P@!�~���O!@)C��!P@1�~���O!@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[27]::TFRecordףp=
�O@!o��!@)ףp=
�O@1o��!@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[26]::TFRecord}x� #�O@!F�>�� @)}x� #�O@1F�>�� @:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[25]::TFRecord�Q�U�O@!Ri���� @)�Q�U�O@1Ri���� @:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[31]::TFRecord�wD��TO@!}g�-� @)�wD��TO@1}g�-� @:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[30]::TFRecordQ���N@!Ii&η� @)Q���N@1Ii&η� @:Advanced file read2�
_Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMapޭ,��~@!��<���P@)���c�(=@1�EvsfK@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[24]::TFRecord���2��?!sX燜]�?)���2��?1sX燜]�?:Advanced file read2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat�ڦx\�c@!�"�o 5@)2��4�?1@{���{?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismY�oC�״?!Ku��P^�?)^h��HK�?1xW�K��v?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchT1��c�?!��C�u?)T1��c�?1��C�u?:Preprocessing2�
hIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap::Shuffle/��0�?!0;�W�i?)/��0�?10;�W�i?:Preprocessing2F
Iterator::Model_����?!�!��Ċ?)�_!se�?1/��_ݘa?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�Q��{��?I �0�;�@Qj̬��.W@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	���;�?u-�
�?����?!���t @	!       "$	�����X@ͯ0�0L�?�����W@!L�g�X@*	!       2$	6�Ou�?/��?87	@�XİØ�?!���9�@:$	o��Z�@��@���?!�f�v�@B	!       J	�x��q��?9��'q�?!��ZӼ��?R	!       Z	�x��q��?9��'q�?!��ZӼ��?b	!       JGPUY�Q��{��?b q �0�;�@yj̬��.W@