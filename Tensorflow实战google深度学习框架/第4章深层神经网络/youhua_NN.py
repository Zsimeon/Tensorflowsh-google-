

#指数衰减法
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)


###########################################################
#在TF中使用tf.exponential_decay函数
global_step = tf.Variable(0)

#通过exponential_decay函数生成学习率
learning_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True)
#因为staircase = True，所以每训练100轮后学习率乘以0.96

#使用指数衰减的学习率，在minimize函数中传入global_step将自动更新
#global_step参数，从而使得学习率也得到更新
learning_step = tf.train.GradientDescentOptimizer(learning_rate) \
    .minimize(...my loss...,global_step = global_step)


#########################
#简单的带L2正则化的损失函数定义
w = tf.Variable(tf.random_normal([2,1], stddev = 1,seed = 1))
y = tf.matmul(x,w)

loss  =tf.reduce_mean(tf.square(y_ - y)) +
 tf.contriblayers.l2_regularizer(lambda)(w)

#########################
weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    #输出为（|1|+|-2|+|-3|+|4|）*0.5=5，其中0.5为正则化的权重
    print sess.run(tf.constrib.layers.l1_regularizer(.5)(weights))
    #输出为(1^2+(-2)^2+(-3)^2+4^2)/2*0.5 = 7.5
    print sess.run(tf.constrib.layers.l2_regularizer(.5)(weights))


#####################################################################
#通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法
#######################
import tensorflow as tf

#获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为‘loss’的集合中
def get_weight(shape, lambda):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合
    #这个函数的第一个参数'losses'是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambda)(var))
    #返回生成的变量
    return var

x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape = (None,1))
batch_size = 8
#定义了每一层网络中节点的个数
layer_dimension = [2,10,10,10,1]
#神经网络的层数
n_layers = len(layer_dimension)

#这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer  = x
# 当前层的节点数
in_dimension = layer_dimension[0]

#通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    #layer_dimension[i]为下一层的节点数
    out_dimension = layer_dimension[i]
    #生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weights = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    #使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
    #进入下一层之前将下一层的节点个数更新为当前层节点个数
    in_dimension = layer_dimension[i]

#在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了图上的集合
#这里只需要计算刻画模型在训练数据上表现得损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

#将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

#get_collection返回一个列表，这个列表是所有这个集合中的元素，在这个样例中，
#这些元素就是损失函数的不同部分，将他们加起来就可以得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))




#################################################################
#滑动平均模型
#ExponentialMovingAverage是如何使用的
import tensorflow as tf

#定义一个变量用于计算滑动平均，这个变量的初始值为0，注意这里手动指定了变量的
#类型为tf.float32,因为所有需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0, dtype = tf.float32)
#这里的step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable = Flase)

#定义一个滑动平均的类（class），初始化时给定了衰减率（0.99）和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
#定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时
#这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    #初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.tun(init_op)

    #通过ema.average(v1)获取滑动平均之后变量的取值，在初始化之后变量v1的值和v1的滑动平均都为0
    print sess.run([v1,ema.average(v1)])        #输出[0.0,0.0]

    #更新变量v1的值到5
    sess.run(tf.assign(v1,5))
    #更新v1的滑动平均值，衰减率为min{0.99,(1+step）/（10+step）=0.1}=0.1
    #所以v1的滑动平均值会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print sess.run([v1,ema.average(v1)])

    #更新step的值为10000
    sess.run(tf.assign(step, 10000))
    #更新v1的值为10
    sess.run(tf.assign(v1,10))
    #更新v1的滑动平均值，衰减率为min{0.99,(1+step）/（10+step）=0.999}=0.99
    #所以v1的滑动平均值会被更新为0.99*4.5+0.01*10=4.555
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])
    #输出[10.0,4.5549998]

    #再次更新滑动平均值，得到的新滑动平均值为0.99*4+0.01*10=4.60945
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])

    #输出[10.0,4.6094499]
    
