import tensorflow as tf 

#定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1,  input2],name="add")

#生成一个写日志的writer，并将当前的Tensorflow计算图写入日志
writer = tf.train.SummaryWriter("/path/to/log",tf.get_default_graph())
writer.close()

#运行tensorBoard，并将日志的地址指向上面程序日志输出的地址
tensorboard --logdir=/path/to/log



###################################

import tensorflow as tf 

with tf.variable_scope("foo"):
	#在命名空间foo下获取变量“bar”，于是得到的变量名称为“foo/bar”
	a = tf.get_variable("bar",[1])
	print(a.name)		#输出：foo/bar:0

with tf.variable_scope("bar"):
	#在命名空间bar下获取变量“bar”，于是得到的变量名称为“bar/bar”
	#此时变量“bar/bar”和变量“foo/bar”并不冲突，可以正常运行
	b = tf.get_variable("bar",[1])
	print(b.name)		#输出：bar/bar:0

with tf.name_scope("a"):
	#使用tf.Variable函数生成变量会受到tf.name_scope影响，于是这个编码的名称为“a/Variable”
	a = tf.Variable([1])
	print(a.name)		#输出：a/Variable:0

	#tf.get_variable函数不受tf.name_scope函数影响
	#于是变量并不在a这个命名空间中
	a = tf.get_variable("b",[1])
	print(a.name)		#输出：b:0

with tf.name_scope("b"):
	#
	#因为tf.get_variable不受tf.name_scope影响，所以这里将试图获取名称
	#为“a”的变量。然而这个变量已经被声明了，于是这里会报重复声明的错误：
	#ValueError:Variable bar already exists, disallowed. Did you mean
	#to set reuse=True in Varscope? Originally defines at:...
	tf.get_variable("b",[1])


############################
#改进9.1节中向量相加的样例代码，使得可视化得到的效果图更加清晰
import tensorflow as tf 

 # 将输入定义放入各自的命名空间中，从而使得TensorBoard可以根据命名空间来整理可视化效果图上的节点
with tf.name_scope("input1"):
 	input1 = tf.constant([1.0,2.0,3.0],name="input1")
with ctf.name_scope("input2"):
 	input2 = tf.Variable(tf.random_uniform([3]),name = "input2")
output = tf.add_n([input1,input2],name = "add")

writer = tf.train.SummaryWriter("/path/to/log",tf.get_default_graph())
