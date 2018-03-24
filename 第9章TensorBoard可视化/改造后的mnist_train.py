import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
#mnist_inference中定义的常量和前向传播的函数不需要改变，因为前向传播已经通过
#tf.variable_scope实现了计算节点按照网络结构的划分
import mnist_inference

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def train(mnist):
	# 将处理输入数据的计算都放在名字为"input"的命名空间下
	with tf.name_scope('input'):
		x = tf.placeholder(
			tf.float32,[None,mnist_inference.INPUT_NODE],name = 'x-input')
		y_ = tf.placeholder(
			tf.float32,[None,mnist_inference.OUTPUT_NODE],
			name = 'y-cinput')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
	y = mnist_inference.inference(x,regularizer)
	global_step = tf.Variable(0, trainable=False)

	#将处理滑动平均相关的计算都放在名为moving_average的命名空间下
	with tf.name_scope('moving_average'):
		variable_average = tf.train.ExponentialMovingAverage(
			MOVING_AVERAGE_DECAY,global_step)
		variables_averages_op = variable_averages.apply(
			tf.trainable_variables())

	#将计算损失函数相关的计算都放在名为loss_function的命名空间下
	with tf.name_scope("loss_function"):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			y, tf.argmax(y_, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

	#将定义学习率、优化方法以及每一轮训练需要执行的操作都放在名字为“train_step”
	#的命名空间下
	with tf.name_scope("train_step"):
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_step,
			mnist.train.num-examples / BATCH_SIZE,
			LEARNING_RATE_DECAY,
			staiecase = True) 
		train_step = tf.train.GradientDescentOptimizer(learning_rate) \ 
				.minimize(loss,global_step=global_step)

		with tf.control_dependencies([train_step,variables_averages_op]):
			train_op = tf.no_op(name='train')

		#使用和5.5节中一样的方式训练神经网络

	#将当前的计算图输出到TensorBoard日志文件
	writer = tf.train.SummaryWriter("/path/to/log",tf.get_default_graph())
	writer.close()

def main(argv = None):
	mnist = input_data.read_data_sets("/tmp/data",one_hot = True)

	train(mnist)

if __name__ == '__main__':
	tf.app.run()
	
