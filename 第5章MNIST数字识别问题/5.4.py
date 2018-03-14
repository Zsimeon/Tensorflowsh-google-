############################
#5.4.1持久化代码实现
import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]),name="v2")
result = v1+v2

init_op = tf.initialize_all_variables()
#声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    #将模型保存到/path/to/model/model.ckpt
    saver.save(sess, "/path/to/model/model.ckpt")

#上面这段代码会生成的第一个文件为model.ckpt.meta，他保存了TF计算图的结构
#第二个文件为model.ckpt，这个文件中保存了TF程序中每一个变量的取值
#最后一个文件为checkpoint文件，保存了一个目录下所有的模型文件列表

#################################################
#以下这段代码给出了加载这个已经保存的TF模型的方法
import tensorflow as tf

#使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]),name="v2")
result = v1+v2

saver = tf.train.Saver()

with tf.Session() as sess:
    #加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, "path/to/model/model.ckpt")
    print(sess.run(result))


#如果不希望重复定义图上的运算，也可以直接加载已经持久化的图
import tensorflow as tf
#直接加载持久化的图
saver = tf.train.import_meta_graph(
    "path/to/model/model.ckpt/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "/path/to/model/model.ckpt")
    #通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
    #输出[3.]


##########################################
#下面这个样例说明变量重命名是如何被使用的

#这里声明的变量名称和已经保存的模型中变量的名称不同
v1 = tf.Variable(tf.constant(1.0, shape=[1]),name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]),name="other-v2")

#如果直接使用tf.train.Saver()来加载模型会报变量找不到的错误

#使用一个字典（dictionary）来重命名变量就可以加载原来的模型了，这个字典指定了原来名称为v1的变量现在加载到变量v1中
#（名称为other-v1），名称为v2的变量加载到变量v2中（名称为other-v2）
saver = tf.train.Saver({"v1":v1,"v2":v2})

#######################################
#以下代码给出一个保存滑动平均模型的样例
import tensorflow as tf
v = tf.Variable(0, dtype = tf.float32, name="v")
#在没有声明滑动平均模型时只有一个变量v，所以下面的语句只会输出“v:0 ”
for variables in tf.all_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.all_variables())
#在声明滑动平均模型后。TF会自动生成一个影子变量
#v/ExponentialMoving Average。于是下的语句会输出
#"v:0"和“v/ExponentialMovingAverage:0”
for variables in tf.all_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(maintain_average_op)
    #保存时，TF会将v:0和v/ExponentialMovingAverage:0两个变量都存下来
    saver.save(sess,"/path/to/model/model.ckpt")
    print(sess.run([v,ema.average(v)]))     #输出[10.0,0.099999905]


#下面代码给出了如何通过变量重命名直接读取变量的滑动平均值。
#下面程序读取的变量v的值实际上是上面代码中v的滑动平均值
v = tf.Variable(0, dtype = tf.float32, name="v")
#通过变量重命名将原来变量v的滑动平均值直接赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "/path/to/model/model.ckpt")
    print(sess.run(v))      #输出0.99999905，这个值就是原来模型中变量v的滑动平均值


####################################
import tensorflow as tf
v = tf.Variable(0,dtype = tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)

#通过使用variables_to_restore函数可以直接生成上面代码中提供的字典
#{“v/ExponentialMovingAverage”:v}
#以下代码会输出：
#{'v/ExponentialMovingAverage':<tensorflow.python.ops.variables.Variable
#object at 0x7ff6454ddc10 >}
#其中后面的Variable类就代表了变量v
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"/path/to/model/model.ckpt")
    print(sess.run(v))      #输出0.099999905，即原来模型中变量v的滑动平均值


##################################################
#TF提供了convert_variables_to_constant函数，通过这个函数可以将计算图中的变量及其取值通过常量的方式保存
import tensorflow as tf
from tensorflow.python.framewrok import graph_util

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result = v1 + v2

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图的Grapher部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    #将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉。
    #在下面的代码中，最后的参数['add']给出了需要保存的节点名称。
    #add节点是上面定义的两个变量相加的操作
    #注意这里给出的是计算节点的名称，所以没有后面的：0
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,graph_def,['add']    )
    #将导出的模型存入文件
    with tf.gfile.GFile("/path/to/model/combined_model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())

######################
#通过下面的程序可以直接计算定义的加法运算的结果
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "/path/to/model/combined_model.pb"
    #读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with tf.gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #将graph_def中保存的图加载到当前的图中
        result = tf.import_graph_def(graph_def,return_elements = ["add:0"])
        print(sess.run(result))



#################################################################
#5.4.2持久化原理及数据格式
################################################################
import tensorflow as tf
#定义变量相加的计算
v1 = tf.Variable(tf.constant(1.0, shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]),name="v2")
result1 = v1 + v2

saver = tf.train.Saver()
#通过export_meta_graph函数导出TF的计算图的元图，并保存为json格式
saver.export_meta_graph("/path/to/model.ckpt.meda.json")



#以下代码展示了如何使用tf.train.NewCheckpointReader类
import tensorflow as tf

#tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointreader('/path/to/model/model.ckpt')

#获取所有变量列表，这是一个从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    #variable_name 为变量名称，all_variables[variable_name]为变量的维度
    print(variables_name, all_variables[variable_name])

#获取名称为v1的变量的取值
print("Value for variable v1 is ", reader.get_tensor("v1"))

'''
这个程序将输出：
v1 [1]              #变量v1的维度为[1]
v2 [1]              #变量v2的维度为[1]
Value for variable v1 is [1.]       #变量v1的取值为1
'''