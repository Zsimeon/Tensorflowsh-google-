###################
#在生成会话时，可以通过设置log_device_placement参数来打印运行每一个运算的设备
##################

import tensorflow as tf 

a = tf.constant([1.0,2.0,3.0], shape=[3],name='a')
b = tf.constant([1.0,2.0,3.0], shape=[3],name='b')
c=  a+b 
#通过log_decive_placement参数来输出运行每一个运算的设备
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

'''
在没有GPU的机器上运行以上代码可以得到以下输出：
Device mapping: no known devices.

add: /job:localhost/replica:0/task:0/cpu:0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
[2. 4. 6.]
'''

'''
Tensorflow程序生成会话时加入了参数log_device_placement=True，
所有程序会将运行每一个操作的设备输出到屏幕
'''

###########################################
import tensorflow as tf 

#通过tf.device将运算指定到特定的设备上

with tf.device('/cpu:0'):
	a = tf.constant([1.0,2.0,3.0], shape=[3],name='a')
	b = tf.constant([1.0,2.0,3.0], shape=[3],name='b')
with tf.device('/gpu:1'):
	c=  a + b 

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

'''
add: /job:localhost/replica:0/task:0/gpu:1
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
[2. 4. 6.]
'''

#################################################

import tensorflow as tf

#在CPU上运行tf.Variable
a_cip = tf.Variable(0, name="a_cpu")

with tf.device('/gpu:0'):
	#将tf.Variable强制放在GPU上
	a_gpu = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	sess.run(tf.initialize_all_variables())

'''
运行上面的程序会报出以下错误：
tensorflow.python.framework.errors.InvalidArgumentError: Cannot assign a device
to node 'a_gpu': Could not satisfy explicit devices specification '/device:GPU:0'
because no supported kernel for GPU deviceis avaliable.
Colocation Dubug Info:
Colocation group had the following types and devices:
Identity:CPU
Assign:CPU
Variable:CPU
[[Node: a_gpu = Variable[container="",dtype = DT_INT32,shape=[],shared_name="",
_device = "/device:GPU:0"]()]]
'''


########################################
#allow_soft_placement参数设置为True时，如果运算无法由GPU执行，
#那么TensorFlow会自动将它放到CPU上执行
#######################################

import tensorflow as tf 

a_cpu = tf.Variable(0, name="a_cpu")
with tf.device('/gpu:0'):
	a_gpu = tf.Variable(0, name="a_gpu")

#通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU上
sess = tf.Session(config=tf.ConfigProto(
	allow_soft_placement=True,log_device_placement = True))
sess.run(tf.initialize_all_variables())

