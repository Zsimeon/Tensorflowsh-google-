#################################################
#创建一个最简单的Tensorflow集群
import tensorflow as tf 
c = tf.constant("Hello,distributes Tensorflow")
#创建一个本地Tensorflow集群
server = tf.train.Server.create_local_server()
#在集群上创建一个会话
sess = tf.Session(server.target)
#输出Hello，distributed Tensorflow
print(sess.run(c))

###################################
#上面的样例代码是只有一个任务的集群，当一个Tensorflow集群有多个任务时，需要
#使用tf.train.ClusterSpec来指定运行每一个任务的机器。
#以下代码展示了在本地运行有两个任务的Tensorflow集群，
####################################
#第一个任务的代码如下
import tensorflow as tf
c = tf.constant("Hello from server1!")

#生成一个有两个任务的集群，一个任务跑在本地2222端口，另外一个跑在本地2223端口
cluster = tf.train.ClusterSpec(
	{"local": ["localhost:2222","localhost:2223"]})
#通过上面生成的集群配置生成Server，并通过job_name和task_index指定当前所启动的任务
#因为该任务是第一个任务，所以task_index为0
server = tf.train.Server(cluster,job_name="local",task_index=0)

#通过server.target生成会话来使用Tensorf集群中的资源，通过设置
#log_device_placement可以看到执行每一个操作的任务
sess = tf.Session(
	server.target,config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
server.join()


###############################
#下面给出了第二个任务的代码
import tensorflow as tf 
c = tf.constant("Hello from server2!")

#和第一个程序一样的集群配置，集群中的每一个任务需要采用相同的配置
cluster = tf.train.ClustreSpec(
	{"local":["localhost:2222","localhost:2223"]})
#指定task_index为1，所以这个程序将在localhost：2223启动服务
server = tf.train.Server(cluster, job_name = "local",task_index=1)
#剩下的代码都和第一个任务的代码一致

###############################################


'''
和使用多GPU类似，TensorFlow支持通过tf.device来指定操作运行在哪个任务上
比如将第二个任务中定义计算的语句改为以下代码，就可以看到这个计算将被调度到
/job:local/replica:0/task:1/cpu:0上面
'''
with tf.device("/job:local/task:1"):
	c = tf.constant("Hello from server2!")


#下面给出了一个比较常见的用于训练深度学习模型的TensorFlow集群配置方法
tf.train.ClusterSpec({
	"worker":[
	"tf-worker0:2222",
	"tf-worker1:2222",
	"tf-worker2:2222"
	],
	"ps":[
	"tf-ps0:2222",
	"tf-ps1:2222"]
	})

