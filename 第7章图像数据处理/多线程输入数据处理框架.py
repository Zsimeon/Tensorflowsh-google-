import tensorflow as tf 

#创建一个先进先出的队列，指定队列中最多可以保存的两个元素，并指定类型为整数
q = tf.FIFOQueue(2,"int32")
#使用enqueue_many函数来初始化队列中的元素，和变量初始化类似
#在使用队列之前需要明确的调用这个初始化过程
init = q.enqueue_many(([0,10],))
#使用Dequeue函数讲队列中的第一个元素出队列。这个元素的值将被存在变量x中
x = q.dequeue()
#将得到的值加1
y = x+1
#将加1后的值再重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
	# 运行初始化队列的操作
	init.run()
	for _ in range(5):
		#运行q_inc将执行数据出队列、出队的元素+1、重新加入队列的整个过程
		v, _ = sess.run([x,q_inc])
		#打印出队元素的取值
		print(v)

'''
队列开始有[0,10]两个元素，第一个出队的为0，加1之后再次入队得到的队列为[10,1]
第二次出队的为10，加1之后入队的为11，得到的队列为[1,11]
以此类推，最后的输出为：
0
10
1
11
2
'''


#########################################
import tensorflow as tf
import numpy as np 
import threading
import time

#线程中运行的程序，这个程序每隔1秒判断是否需要停止并打印自己的ID
def MyLoop(coord,worker_id):
	#使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
	while not coord.should_stop():
		#随机停止所有的线程
		if np.random.rand() < 0.1:
			print("Stoping from id: %d \n" % worker_id,)
			#调用coord.request_stop()函数来通知其他线程停止
			coord.request_stop()
		else:
			#打印当前线程的id
			print("Working on id: %d\n"% worker_id,)
		#暂停1秒
		time.sleep(1)

#声明一个tf.train.Coordinator类来协同多个线程
coord = tf.train.Coordinatior()
#声明创建5个线程
threads = [
	theading.Thread(target=MyLoop,args = (coord,i,)) for i in range(5)]
#启动所有的线程
for t in threads: t.start()
#等待所有线程退出
coord.join(threads)

'''
上面的运行结果类似为
Working on id: 0
Working on id: 1
Working on id: 2
Working on id: 3
Working on id: 4
Working on id: 0
Stoping from id: 4
Working on id: 1
'''

#############################
#使用tf.train.QueueRunner()和tf.train.Coordinator来管理多线程队列操作
import tensorflow as tf

#声明一个先进先出的队列，队列中最多100个元素，类型为实数
queue = tf.FIFOQueue(100,"float")
#定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

'''
使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
tf.train.QueueRunner的第一个参数给出了被操作的队列，[enqueue_op]*5
表示了需要启动的5个线程，每个线程中运行的是enqueue_op操作
'''
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

'''
将定义过的QueueRunner加入TensorFlow计算图上指定的集合
tf.train.add_queue_runner函数没有指定集合
则加入默认集合tf.GraphKeys.QUEUE_RUNNERS
下面的函数就是将刚刚定义的qr加入默认的tf.GraphKeys.QUEUE_RUNNERS集合
'''
tf.train.add_queue_runner(qr)
#定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
	#使用tf.train.Coordinator来协同启动的线程
	coord = tf.train.Coordinator()
	'''
	使用tf.train.QueueRunner时，需要明确调用tf.train.start_queue_runners来启动线程
	否则因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队操作被运行
	tf.train.start_queue_runners函数会默认启动tf.GraphKeys.QUEUE_RUNNERS集合总所有的
	QueueRunner。因为这个函数只支持启动指定集合中的QueueRunner，所以一般来说
	tf.train.add_queue_runners函数和tf.train.start_queue_runners函数会指定同一个集合
	'''
	tfhreads = tf.train.start_queue_runners(sess = sess,coord = coord)
	#获取队列中的取值
	for _ in range(3):
		print(sess.run(out_tensor))[0]

	#使用tf.train.Coordinator来停止所有的线程
	coord.request_stop()
	coord.join(threads)


'''
上面的程序将启动五个线程来执行队列入队操作，其中每一个线程都是将随机数写入队列
于是在每次运行出队操作时，可以得到一个随机数。可以得到类似下面的结果
-0.315963
-1.06425
0.347479
'''
