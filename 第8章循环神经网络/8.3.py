########################################
#MultiRNNCell类来实现深层循环神经网络的前向传播过程
########################################


#定义一个基本的额LSTM结构作为循环体的基础结构，深层循环神经网络也支持使用其他的循环体结构
lstm = rnn_cell.BasicLSTMCell(lstm_size)
#通过MultiRNNCell类实现深层循环神经网络中每一个时刻的前向传播过程
#其中number_of_layers表示有多少层，也就是从x_t到h_t需要经过多少个LSTM结构
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

#通过zero_state函数来获取初始状态
state = stacked_lstm.zero_state(batch_size,tf.float32)

#计算每一时刻的前向传播结果
for i in range(len(num_steps)):
	if i>0: tf.get_variable_scope().reuse_variables()
	stacked_lstm_output, state = stacked_lstm(current_input,state)
	final_output = fully_connected(stacked_lstm_output)
	loss += calc_loss(final_output,ecpected_output)


#################################
#使用tf.nn.rnn_cell.DropoutWrapper实现dropout功能
#################################

#定义LSTM结构
lstm = rnn_cell.BasicLSTMCell(lstm_size)

#使用DropoutWrapper类来实现dropout功能。该类通过两个参数来控制dropout的概率
#一个参数为input_keep_prob，它可以用来控制输入的dropout概率
#另一个为output_keep_prob，可以用来控制输出的dropout概率
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob)

#在使用了dropout的基础上定义
stacked_lstm = rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)

