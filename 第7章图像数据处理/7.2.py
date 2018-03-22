################
#tensorflow对jpeg格式图像的编码/解码过程

import matplotlib.pyplot as plt
import tensorflow as tf

#读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("/path/to/picture",'r').read()

with tf.Session() as sess:
	#将图像使用jpeg的格式解码从而得到图像对应的三维矩阵，TensorFlow
	#还提供了tf.iamge.decode_png函数对png格式的图像进行解码。
	#解码之后的结果为一个张量，在使用它的取值之前需要明确调用运行的过程
	img_data = tf.image.decode_jpeg(image_raw_data)

	print(image_data.eval())
	#输出解码之后的三维矩阵，上面这一行将输出下面的内容
	'''
	[[[165 160 138]
	...,
	[105 140 50]]
	[[166 161 139]
	...,
	[[207 200 181]
	...,
	[106 81 50]]]
	'''

	#使用pyplot工具可视化得到的图像
	plt.imshow(img_data.eval())
	plt.show()

	#将数据的类型转化为实数方便下面的样例程序对图像进行处理
	img_data = tf.image.convert_image_dtype(img_data,dtype = tf.float32)

	#将表示一种图像的三维矩阵重新按照jpeg格式编码并存入文件。
	#打开这种图像，可以得到和原始图像一样的图像
	encoded_image = tf.image.encode_jpeg(img_data)
	with tf.gfile.GFile("/path/to/output","wb") as f:
		f.write(encoded_image.eval())



###########################
#图像大小调整  tf.image.resize_images
###########################

#加载原始图像，定义会话等过程和图像编码处理中代码一致
#假设img_data是已经解码且进行过类型转化的图像

#通过tf.image.resize_images函数调整图像的大小。这个函数第一个参数为原始图像。
#第二个第三个参数为调整后图像的大小，method参数给出了调整图像大小的算法
resized = tf.image.resize_images(img_data,300,300,method = 0)

#输出调整后的图像的大小，此处的结果为（300,300，？），表示图像大小为300*300
#但图像深度在没有明确设置之前是个问号
print(img_data.get_shape())

'''
0 -- 双线性插值法
1 -- 最近邻居法
2 -- 双三次插值法
3 -- 面积插值法
'''

#tf.image.resize_image_with_crop_or_pad()函数调整图像大小
#第一个参数为原始图像，后面两个参数是调整后的目标图像大小
#如果原始图像尺寸大于目标图像，那么这个函数会自动截取原始图像中居中的部分
#如果目标图像大于原始图像，这个函数会在原始图像的四周填充全0背景。
croped = tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
padded = tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)


#按比例裁剪图像
central_cropped = tf.image.central_crop(img_data,0.5)


################################################
#图像翻转
################################################

#将图像上下翻转
flipped = tf.image.flip_up_down(img_data)
#将图像左右翻转
flipped = tf.image.random_flip_left_right(img_data)
#将图像沿对角线翻转
transposed = tf.image.transpose_image(img_data)


#以一定概率上下翻转图像
flipped = tf.image.random_flip_up_down(img_data)
#以一定概率左右翻转图像
flipped = tf.image.random_flip_left_right(img_data)


#####################################
# 图像色彩调整

#将图像的亮度-0.5
adjusted = tf.image.adjust_brightness(img_data,-0.5)
#将图像的亮度+0.5
adjusted = tf.image.adjust_brightness(img_data,0.5)

#在[-max_delta,max_delta]的范围随机调整图像的亮度
adjusted = tf.image.random_brightness(image,max_delta)



#################################
#调整图像对比度

#将图像对比度-5
adjusted = tf.image.adjust_contrast(img_data,-5)
#将图像对比度+5
adjusted = tf.image.adjust_contrast(img_data,5)

#在[lower,upper]的范围随机调整图像的对比度
adjusted = tf.image.random_contrast(image,lower,upper)



####################################
#调整图像的色相
adjusted = tf.image.adjust_hue(img_data,0.1)

#在[-max_delta,max_delta]的范围随机调整图像的色相
#max_delta的取值子啊[0,0.5]之间
adjusted = tf.image.random_hue(image,max_delta)

####################################
#调整图像的饱和度
#将饱和度-5
adjusted = tf.image.adjust_saturation(img_data,-5)
adjusted = tf.image.adjust_saturation(img_data,5)
#在[lower,upper]的范围随机调整图的饱和度
adjusted = tf.image.random_saturation(image,lower,upper)


#将代表一张图像的三维矩阵中的数字均值变为0，方差变为1
adjusted = tf.image.per_image_whitening(img_data)


###########################
#添加标注框

#将图像缩小一些，这样可视化能让标注框更加清楚
img_data = tf.image.resize_images(img_data,180,267,method = 1)
#tf.image.draw_bounding_boxes()函数要求图像矩阵中的数字为实数，
#先将图像矩阵转化为实数类型，函数输入是一个batch的数据，也就是多张图像组成的
#四维矩阵，所以需要将解码之后的图像矩阵加一维
batched = tf.expand_dims(
	tf.image.convert_image_dtype(img_data,tf.float32),0)
#给出每一张图像的所有标注框，一个标珠框有四个数字，分别为
#[y_min,x_min,y_max,x_max]
boxes =tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.447,0.5,0.56]]])
result = tf.image.draw_bounding_boxes(batched,boxes)




################################
#通过tf.image.sample_distorted_bounding_box()函数来随机截取图像
boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.447,0.5,0.56]]])
#可以通过提供标注框的方式来告诉随机截取图像的算法哪些部分是“有信息量的”
begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
	tf.shape(img_data),bounding_boxes=boxes)

#通过标注框可视化随机截取得到的图像
batched = tf.axpand_dims(
	tf.image.convert_image_dtype(img_data,tf.float32),0)
image_with_box = tf.image.draw_bounding_boxes(batched,bbox_for_draw)

#截取随机出来的图像
distored_image = tf.slice(img_data,begin,size)


#########################################
#图像预处理完整样例
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
给定一张图像，随机调整图像的色彩
'''
def distort_color(image,color_ordering=0):
	if color_ordering ==0:
		image = tf.image.random_brightness(image,max_delta=32./225.)
		image = tf.image.random_saturation(image,lower = 0.5,upper = 1.5)
		image = tf.image.random_hue(image,max_delta = 0.2)
		image = tf.image.random_contrast(iamge,lower = 0.5,upper = 1.5)
	elif color_ordering == 1:
		image = tf.image.random_saturation(image,lower = 0.5,upper = 1.5)
		image = tf.image.random_brightness(image,max_delta=32./225.)
		image = tf.image.random_contrast(iamge,lower = 0.5,upper = 1.5)
		image = tf.image.random_hue(image,max_delta = 0.2)
	elif color_ordering == 2:
		...
	return tf.clip_by_value(image,0.0,1.0)

'''
给定一张解码后的图像、目标图像的尺寸以及图像上的标注框，此函数可以对给出的图像进行预处理
这个函数的输入是原始的训练图像，输出则是神经网络模型的输入层
'''
def preprocess_for_train(image,height,width,bbox):
	# 如果没有提供标注框，则认为整个图像就是需要关注的部分
	if bbox is None:
		bbox = tf.constant([0.0,0.0,1.0,1.0],
			dtype = tf.float32,shape=[1,1,4])
	#转换图像张量的类型
	if image.dtype != tf.float32:
		iamge = tf.image.convert_image_dtype(image,dtype = tf.float32)

	#随机截取图像，减少需要关注的物体大小对图像认识算法的影响
	bbox_begin,bbox_size,_ = tf.iamge.sample_distorted_bounding_box(
		tf.shape(image),bounding_boxes=bbox)
	distorted_image = tf.slice(image,bbox_begin,bbox_size)
	#将随机截取的图像调整为神经网络输入层的大小
	distorted_image = tf.image.resize_images(
		distorted_image, height,width,method=np.random.randint(4))
	#随机左右翻转图像
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	#使用一种随机的顺序调整图像色彩
	distorted_image = dietort_color(distorted_image,np.random.randint(2))
	return distorted_image

image_raw_data = tf.gfile.FastGFile("/path/to/picture","r").read()
with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])

	#运行6次获得6种不同的图像，
	for i in range(6):
		#将图像尺寸调整为299*299
		result = preprocess_for_train(img_data,299,299,boxes)
		plt.imshow(result.eval())
		plt.show()