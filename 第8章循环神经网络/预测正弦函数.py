# -*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

#加载matplotlib工具包，使用该工具可以对预测的sin函数曲线进行绘图
import matplotlib as mpl 
mpl.use('Agg')
from matplotlib import pyplot as plt 

learn = tf.contrib.learn 

HIDDEN_SIZE = 30	#LSTM中隐藏节点的个数
NUM_LAYERS = 2	#LSTM的层数

TIMESTEPS = 10		#循环神经网络的截断长度
TRAINING_STEPS = 10000		#训练轮数
BATCH_SIZE = 32				#batch的大小

TRAINING_EXAMPLES = 10000	#训练数据个数
TESTING_EXAMPLES = 1000		#测试数据个数
SAMPLE_GAP = 0.01			#采用间隔

def generate_data(seq):
	X = []
	y = []
	#序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i+TIMESTEPS项作为输出
	#即用sin函数前面的TIMESTEPS个点的信息，预测第i+TIMESTEPS个点的函数值
	for i in range(len(seq) - TIMESTEPS -1):
		X.append([seq[i: i+TIMESTEPS]])
		y.append([seq[i+TIMESTEPS]])
	return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y):
	#使用多层的lstm结构
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
	cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
	x_ = tf.unpack(X, axis = 1)

	#使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果
	output,_ = tf.nn.rnn(cell, x_, dtype = tf.float32)
	#在本问题中只关注最后一个时刻的输出结果，该结果为下一时刻的预测值
	outptu = output[-1]

	#对LSTm网络的输出再做加一层全连接层并计算损失。注意这里默认的损失为平均平方差损失函数
	prediction, loss = learn.models.linear_regression(output,y)

	#创建模型优化器并得到优化步骤
	train_op = tf.contrib.layers.optimize_loss(
		loss, tf.contrib.framework.get_global_step(),
		optimizer="Adagrad", learning_rate = 0.1)

	return prediction,loss,train_op

#建立深层循环网络模型
regressor = learn.Estimator(model_fn=lstm_model)

#用正弦函数生成训练和测试数据集合
#numpy.linspace函数可以创建一个等差序列的数组，它常用的参数有三个参数，第一个参数表示起始值，
#第二个参数表示终止值，第三个参数表示数列的长度
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sim(np.linspace(
	0, test_start, TRAINING_EXAMPLES,dtype = np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
	test_start, test_end, TESTING_EXAMPLES, dtype = np.float32)))

#调用训练好的模型对测试数据进行预测
predicted = [[pred] for pred in regressor.predict(test_X)]
#计算rmse作为评价指标
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis = 0))
print("Mean Square Error is: %f" % rmse[0])


#对预测的sin函数曲线进行绘图，并存储到运行目录下的sin.png
fig = plt.figure()
plot_predicted = plt.plot(predicted,label='predicted')
plot_test = plt.plot(test_y,label='real_sin')
plt.legend([plot_predited,plot_test],['predicted','real_sin'])

fig.savefig('sin.png')

