?	R???J@R???J@!R???J@	č?n????č?n????!č?n????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLR???J@??Os??@1zlˀ?x0@A?p̲'???Iޭ,?Yr>@Y??^~????rEagerKernelExecute 0*	??C???@2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::MemoryCacheImpl::ParallelMapV2ѓ2??M@!??gn?U@)ѓ2??M@1??gn?U@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::MemoryCacheImplv??=?@!???;JeV@)?4LkӴ?1!??z?@:Preprocessing2?
qIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::MemoryCacheImpl::ParallelMapV2::Shuffle?lu9% ??!?>?@)?YO????1??'? &@:Preprocessing2?
~Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::MemoryCacheImpl::ParallelMapV2::Shuffle::TensorSlice??8??¦?!*t?|????)??8??¦?1*t?|????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2	]ޜ@!?S??,W@)??X?????1?M??t???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???????!'5??@)e#?#Ԥ?1t?=?T???:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat??z2?H@!|?<Kp?V@)????k???1??]?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchJΉ=????!?!???y??)JΉ=????1?!???y??:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::MemoryCache???Z?@!???/?V@)ޒ??ɓ?1H{??ya??:Preprocessing2F
Iterator::Model??@?S??!?Q<??@)???'?.??1B?h`???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?57.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9č?n????ITW})w*Q@QvP?H?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Os??@??Os??@!??Os??@      ??!       "	zlˀ?x0@zlˀ?x0@!zlˀ?x0@*      ??!       2	?p̲'????p̲'???!?p̲'???:	ޭ,?Yr>@ޭ,?Yr>@!ޭ,?Yr>@B      ??!       J	??^~??????^~????!??^~????R      ??!       Z	??^~??????^~????!??^~????b      ??!       JGPUYč?n????b qTW})w*Q@yvP?H?@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?蛾???!?蛾???0"&
CudnnRNNCudnnRNN?p??b???!6!?0?J??"D
)sequential/embedding/embedding_lookup/_11_Send?v!B????!?,?????";
gradients/split_2_grad/concatConcatV2'???q?!8?<M??"*
transpose_0	TransposeGX{zc{o?!?^???#??"(

concat_1_0ConcatV2ti?V?g?!??M?:??"9
gradients/split_grad/concatConcatV2)0c?5?b?!)??~M??";
gradients/split_1_grad/concatConcatV2?^?f?Wb?!?!!?_??" 
splitSplitm?x??\?!??(x+n??"*
transpose_9	Transpose?$;?BW?!]_%{?y??Q      Y@Y:`??Z)@a??	??U@q???^~@y/??UN??"?

both?Your program is POTENTIALLY input-bound because 11.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?57.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 