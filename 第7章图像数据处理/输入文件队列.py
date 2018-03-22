 import tensorflow as tf

 #创建TFRecord文件的帮助函数
 def _int64_feature(value):
 	return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

 	#模拟海量数据情况下将数据写入不同的文件。num_shards定义了总共写入多少个文件
 	#instances_per_shard定义了每个文件中有多少个数据
 num_shards = 2
 instances_per_shard = 2
 for i in range(num_shards):
 	'''
 	将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分
 	m表示了数总共被存在多少个文件中，n表示当前文件的编号。
 	'''
 	filename = ('/path/to/data.tfrecords-%.5d-of-%.5d' % (i,num_shards))
 	writer = tf.python_io.TFRecordWriter(filename)
 	#将数据封装成Example结构并写入TFRecord文件
 	for j in range(instances_per_shard):
 		#Example结构仅包括当前样例属于第几个文件以及是当前文件的第几个样本
 		example = tf.train.Example(features = tf.train.Features(feature ={
 			'i': _int64_feature(i),
 			'j': _int64_feature(j) 			}))
 		writer.write(example.SerializeToString())
 	writer.close()

'''
程序运行之后，在指定的目录下将生成两个文件：/path/to/data.tfrecords-00000-of-00002
和/path/to/data.tfrecords-00001-of-00002.每一个文件中存储了两个样例。
'''

#################################
#以下代码展示了tf.train.match_filenames_once函数和tf.train.string_input_prodicer
#函数的使用方法

import tensorflow as tf 
#使用tf.train.match_filenames_once函数获取文件列表
files = tf.train.match_filenames_once("/path/to/data.tfrecords-*")

'''
通过tf.train.string_input_prodicer函数创建输入队列，输入队列中的文件列表为
tf.train.match_filenames_once函数获取的文件列表。
这里将shuffle参数设为False来避免随机打乱读文件的顺序
但一般在解决实际问题时，会将shuffle参数设为True
'''
filename_queue = tf.train.string_input_producer(files,shuffle = False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
	serialized_example,
	features = {
	'i':tf.FixedLenFeature([],tf.int64),
	'j':tf.FixedLenFeature([],tf.int64),
	})

with tf.Session() as sess:
	# 虽然在本段程序里没有声明变量，但使用tf.train.match_filenames_once函数时
	#需要初始化一些变量
	tf.initialize_all_variables().run()
	'''
	打印文件列表将得到下面的结果：
	['/path/to/data.tfrecords-00000-of-00002'
	 '/path/to/data.tfrecords-00001-of-00002']
	'''
	print(sess.run(siles))

	#声明tf.train.Coordinator类来协同不同线程，并启动线程
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess = sess, coord = coord)

	#多次执行获取数据的操作
	for i inrange(6):
		print(sess.run([features['i'],features['j']]))
	coord.request_stop()
	coord.join(threads)

'''
上面的打印将输出
[0,0]
[0,1]
[1,0]
[1,1]
[0,0]
[0,1]
'''

################################
#组合训练数据
import tensorflow as tf

example, label = features['i'],features['j']

batch_size = 3

capacity = 1000 + 3 * batch_size

example_batch,label_batch = tf.train.batch(
	[example,label],batch_size = batch_size,capacity = capacity)

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	for i in rang(2):
		cur_example_batch,cur_label_batch = sess.run(
			[example_batch,label_batch])
		print(cur_example_batch,cur_label_batch)

	coord.request_stop()
	coord.join(threads)

'''
运行上面的程序可以得到下面的输出
[0 0 1] [0 1 0]
[1 0 0 ][1 0 1]
'''



######################################################
#输入数据处理框架
import tensorflw as tf 

#创建文件列表，并通过文件列表创建输入文件队列。在调用输入数据处理流程前，需要统一
#所有原始数据的格式并将它们存储到TFRecord文件中。
files = tf.train.match_filenames_once("/path/to/file_pattern-*")
filename_queue = tf.trian.string_input_producer(files, shuffle = False)

#解析TFRecord文件里的数据
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
	serialized_example,
	features = {
	'image':tf.FixedLenFeature([],tf.string),
	'label':tf.FixedLenFeature([],tf.nit64),
	'height':tf.FixedLenFeature([],tf.int64),
	'width',tf.FixedLenFeature([],tf.int64),
	'channels':tf.FixedLenFeature([],tf.int64),})
#假设image存储的是图像数据，label为该样例所对应的标签
#height、width、channels给出了图片的维度
image, label = features['image'],features['label']
height,width = features['height'],features['width']
channels = features['channels']

#从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image,tf.uint8)
decoded_image.set_shape([height,width,channels])
#定义神经网络输入层图片的大小
image_size = 299
#preprocess_for_train为前面介绍的图像预处理程序
distorted_image = preprocess_for_train(
	decoded_image, image_size, image_size,None)

#将处理后的图像和标签数据通过tf.train.shuffle_batch整理成神经网络训练时需要的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
	[distorted_image, label], batch_size = batch_size,
	capacity = capacity, min_after_dequeue = min_after_dequeue)

#定义神经网络的结构以及优化过程。image_batch可以作为输入提供给神经网络输入层
#label_batch则提供了输入batch中样例的正确答案
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                     .minimize(loss)

#声明会话并运行神经网络的优化过程
with tf.Session() as sess:
	#神经网络训练准备工作。这些工作包括变量初始化、线程启动
	tf.initialize_all_variables().run()
	coord = tf.train.Coordinator()
	thraads = tf.train.start_queue_runners(sess=sess, coord=coord)

	#神经网络训练过程
	for i in range(TRAINING_ROUNDS):
		sess.run(train_step)

	#停止所有线程
	coord.request_stop()
	coord.join(threads)
	

