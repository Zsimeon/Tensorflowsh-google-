from tensorflow.examples.tutorials.mnist import input_data

#载入MNIST数据集，如果指定地址/path/to/MNIST_data 下没有已经下载好的数据
#那么Tensorflow会自动从网络下载
mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot = True)

#打印Training Data  size： 55000
print "Training data size:", mnist.train.num_examples

#打印ValidatingData size：5000
print("Validating data size:", mnist.validation.num_examples)

#打印Testing data size：10000
print("Testing data size:", mnist.test.num_examples)

#打印Example training data :[0. 0. 0. ... 0.380 0.376 .. 0.]
print("Example training data:", mnist.train.images[0])

#打印Example training data label：
#[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. ]
print("Example training data label:", mnist.train.labels[0])

#################################################
#mnist.train.next_batch函数，可以从所有的训练数据中读取一小部分作为一个训练batch
batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
#从train的集合中选取batch_size个训练数据
print("X shape:", xs.shape)
#输出Xshape：(100,784)
print("Y shape", ys.shape)
#输出Y的shape：（100,10）
##############################################

