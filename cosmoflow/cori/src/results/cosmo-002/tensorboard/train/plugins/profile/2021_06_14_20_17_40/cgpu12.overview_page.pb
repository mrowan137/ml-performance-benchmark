�&$	]��D	�Z@`D\h���?a��qBZ@!�"���Z@	�{J"��?��^5T�?!	f����?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8a��qBZ@÷�n�;�?1}(2X@A���X�?Ib�� ��@Y�@��?"y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8����SZ@/�HM���?1���VX@A��
DOz@I�a��h#�?Y�:���;�?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/�"���Z@���5W@1���¬X@A��w.�?IX�Q�@*q=
W��AS�*PA2�
VIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2����d@!�{9�r"8@)����d@1�{9�r"8@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2���q<U@!nmҼP�(@)'��9U@1�k�x�(@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[28]::TFRecord�<֌*P@!��̝�"@)�<֌*P@1��̝�"@:Advanced file read2�
_Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap�	ܺۂ{@!(��Q��O@)qh�P@1��Ol�"@:Preprocessing2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[27]::TFRecord�ܵ�|�O@!3�� j"@)�ܵ�|�O@13�� j"@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[26]::TFRecordIc���RO@!�
��"@)Ic���RO@1�
��"@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[30]::TFRecord#�-�RO@!�&��y�!@)#�-�RO@1�&��y�!@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[31]::TFRecordV��L�N@!��H>M�!@)V��L�N@1��H>M�!@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[29]::TFRecord	Q���N@!Ϫ�d�!@)	Q���N@1Ϫ�d�!@:Advanced file read2�
mIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap[25]::TFRecord
�����?!+��<�0�?)
�����?1+��<�0�?:Advanced file read2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat�vi���d@!=���|$8@)]��u?�?1S��rP�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismux�q�?!�֟(�?)� Ϡ��?1��ߗ�K{?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchA����A�?!z�ͧ��z?)A����A�?1z�ͧ��z?:Preprocessing2�
hIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteRepeat::ParallelMapV2::FlatMap::ShuffleI�Ǵ6��?!x5��q?)I�Ǵ6��?1x5��q?:Preprocessing2F
Iterator::Model�LLb��?!��F�%�?)�Q���?1v��j��d?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"�4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9[`���?I��x� @Q�~�%)�V@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	h�s݁��?ڐ���?/�HM���?!���5W@	!       "$	��OAkSX@���\��?���VX@!���¬X@*	!       2$	9jr;u@���F��@���X�?!��
DOz@:$	��KT@��I��
@�a��h#�?!X�Q�@B	!       J	�)�"�?�����?!�@��?R	!       Z	�)�"�?�����?!�@��?b	!       JGPUY[`���?b q��x� @y�~�%)�V@�"-
IteratorGetNext/_1_Send�k_�&��?!�k_�&��?"m
=gradient_tape/sequential/conv3d/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2�J&���?!T�T_�?�?08"-
IteratorGetNext/_2_Recv�� �rW�?!�*]���?"6
sequential/conv3d/Conv3DConv3D�'Z�q��?!G��K���?0"a
>gradient_tape/sequential/max_pooling3d/MaxPool3D/MaxPool3DGradMaxPool3DGrad��!lډ�?!^�|�?"_
<gradient_tape/sequential/leaky_re_lu/LeakyRelu/LeakyReluGradLeakyReluGrad�T�'��?!h�ۻ��?"o
?gradient_tape/sequential/conv3d_1/Conv3D/Conv3DBackpropFilterV2Conv3DBackpropFilterV2n��yȱ�?!�7xC��?08"m
>gradient_tape/sequential/conv3d_1/Conv3D/Conv3DBackpropInputV2Conv3DBackpropInputV2��9�`B�?!m֫N+��?08"6
sequential/conv3d/BiasAddBiasAddj\�l��?!P�D����?"A
"sequential/max_pooling3d/MaxPool3D	MaxPool3D��s�[^�?!�g����?I���P�C@Qzv��6N@Y��p,@a߅��qU@qp���6��?y*�����Q?"�	
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 