$	�MʷQZ@~Cgm{0�?�(���Y@!����B.Z@	@sR���?�H��+T�?!�\ڿA��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8����B.Z@I���p�@1�i�q�X@A�TO�}�?I\������?Yi��U��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8�Tl��Z@�5Z���?1��֪�W@AP��0{Y@I�y8�)�?Y�'-\Va�?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/	�(���Y@4��X�@1���d�W@A6��x"��?I'���@*��N|+A��o$�$A)      �=2�
VIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2�[�T�@!%E$�|C@)�[�T�@1%E$�|C@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[28]::TFRecord@��
^f@!��	�Е&@)@��
^f@1��	�Е&@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[23]::TFRecord���o��d@!5T�%@)���o��d@15T�%@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[22]::TFRecord.v��2V]@!li�YU�@).v��2V]@1li�YU�@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[25]::TFRecord)�A&1\@!�[�"aw@))�A&1\@1�[�"aw@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[27]::TFRecord.�|���Y@!����0@).�|���Y@1����0@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[24]::TFRecord�Y�$|X@!�C(=/�@)�Y�$|X@1�C(=/�@:Advanced file read2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2�UfJkh@!%�!�}�(@)���\S@1 ���@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[26]::TFRecord?T1�1O@!�/Gw@)?T1�1O@1�/Gw@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[29]::TFRecord�ȯ0N@!�D��a{@)�ȯ0N@1�D��a{@:Advanced file read2�
_Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap;�d*�@!��P�pL@)N�#Ed�?1Y@�	��?:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[21]::TFRecord��o_�?!!���;R�?)��o_�?1!���;R�?:Advanced file read2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat����@!A57C@)NG 7��?1-�|Qw?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismA�º�?!��N��*w?)��<+iŧ?1]A�� h?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�U�Z�?!ND�/Uf?)�U�Z�?1ND�/Uf?:Preprocessing2�
hIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap::Shuffle	��'��?!�2r԰a?)��'��?1�2r԰a?:Preprocessing2F
Iterator::Model�O9&��?!�&���{?)�~�٭e�?1e��슓R?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�[�/5�?I�O����@Qz����W@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	i���A|	@kKga��@�5Z���?!I���p�@	!       "$	�3��X@qƏ˱��?���d�W@!�i�q�X@*	!       2$	�$�[@�m���#@�TO�}�?!P��0{Y@:$	ǋrV1@ߖv�!�@�y8�)�?!'���@B	!       J	�� �rh�?p,����?!i��U��?R	!       Z	�� �rh�?p,����?!i��U��?b	!       JGPUY�[�/5�?b q�O����@yz����W@