import numpy as np

X = [1,2]
state = [0.0,0.0]
#分开定义不同输入部分的权重以方便操作
w_cell_staet = np.asarray([[0.1,0.2],[0.3,0.4]])
w_call_input = np.asarray([0.5,0.6])
b_cell = np.asarray([0.1,-0.1])

#定义用于输出的全连接层参数
w_output = np.asarray([[1.0],[2.0]])
b_output = 0.1

#按照时间顺序执行循环神经网络的前向传播过程
for i in range(len(X)):
	#计算循环体重的全连接层神经网络
	before_activation = np.dot(state, w_cll_stae)+X[i]*w_cell_input + b_cell
	state = np.tanh(before_activation)

	#根据当前时刻状态计算最终输出
	fianl_output = np.dot(state,w_output)+b_output

	#输出每个时刻的信息
	print("before activation:",before_activation)
	print("state:",state)
	print("output:",final_output)


'''
运行以上程序可以得到输出：
before activation:[0.6 0.6]
state: [0.53704957 0.46211716]
output:[1.56128388]
before activation:[1.2923401 1.39225678]
state:[0.85973818 0.8836641]
output:[2.72707101]
'''



#################################################
#Tensorflow中实现使用LSTM结构的循环神经网络的前向传播过程


#定义一个LSTM结构
#LSTM中使用的变量也会在该函数中自动被声明
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)

'''
将LSTM中的状态初始化为全0数组。在优化循环神经网络时，每次也会使用一个batch的训练样本
'''
state = lstm.zero_state(batch-size,tf.float32)

#定义损失函数
loss = 0.0

#虽然在理论上循环神经网络可以处理任意长度的序列，但是在训练时为了避免梯度消散的问题
#会规定一个最大的序列长度，以下代码中，用num_steps来表示这个长度
for i in range(num_steps):
	#在第一个时刻声明LSTM结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
	if i > 0: tf.get_variable_scope().reuse_variables()

	#每一步处理时间序列中的一个时刻。将当前输入和前一时刻状态（state）
	#传入定义的LSTM结构可以得到当前LSTM结构的输出lstm_output和更新后的状态state
	lstm_output,state = lstm(current_input,state)
	#将当前时刻LSTM结构输出传到一个全连接层得到最后的输出
	final_output = fully_conected(lstm_output)
	#计算当前时刻输出的损失
	loss += calc_loss(final_output,expected_output)

#使用类似第4章中介绍的方法训练模型
