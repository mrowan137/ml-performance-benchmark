$	˪�k�/[@��u��
 @���>țZ@!v�[@$	��j��?��p�j	�?_��p��?!���$���?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8v�[@��*P��@1���N�.Y@A-B�4-�?Is�]��?@Y�w)u�8�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8	����t[@��J�?1��d�z2Y@A\='�ol@IKZ����?Y�W歺�?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8
���>țZ@(}!�@1b,�/X@AįX�E�?I(���@Y��|����?*��"�dB+A�I)�'A2�
VIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2	�zj��@!���M�QB@)�zj��@1���M�QB@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2���Yν�@!��K���D@)W���b@1�A���!@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[27]::TFRecord�~���`@!�U4* @)�~���`@1�U4* @:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[28]::TFRecord*�TPQ�`@!A�t���@)*�TPQ�`@1A�t���@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[22]::TFRecord�2�F�D`@!Pd�I@)�2�F�D`@1Pd�I@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[24]::TFRecorda��*/Z@!���-@)a��*/Z@1���-@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[21]::TFRecord�c�]K�X@!ufJ���@)�c�]K�X@1ufJ���@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[23]::TFRecord]lZ)�X@!2�l���@)]lZ)�X@12�l���@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[29]::TFRecord��d�'P@!���w@)��d�'P@1���w@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[26]::TFRecord����r&P@!���"@)����r&P@1���"@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[25]::TFRecord����qP@!JC�$o�@)����qP@1JC�$o�@:Advanced file read2�
_Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap	��g��H�@!��ky2K@)��fX(@1Vi/�h�?:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[20]::TFRecord������?!�D{�?)������?1�D{�?:Advanced file read2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat	�!�u��@!�0
�^RB@)#/kb��?1$�BO�rz?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�g����?!��n��t?):3P��?1F̏��f?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchW횐��?!�+�!!Sc?)W횐��?1�+�!!Sc?:Preprocessing2�
hIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap::Shuffle
�N��唠?!Z�l���_?)�N��唠?1Z�l���_?:Preprocessing2F
Iterator::Model��r��{�?!�)�%�x?)n�\p�?1�k)�,IN?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9o� J~�?IPR�~� @Q�T��V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	˧ǶX	@��e�*�@��J�?!��*P��@	!       "$	� `�X@brDJx@b,�/X@!��d�z2Y@*	!       2$	+6�uā@OO��x@-B�4-�?!\='�ol@:$	�$$�6�@B x@KZ����?!(���@B	!       J$	��'��?�r���?��|����?!�w)u�8�?R	!       Z$	��'��?�r���?��|����?!�w)u�8�?b	!       JGPUYo� J~�?b qPR�~� @y�T��V@