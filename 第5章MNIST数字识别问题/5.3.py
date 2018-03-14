
##########################
#以下代码给出了通过这两两个函数创建同一个变量的样例
##########################
#下面这两个定义是等价的
v = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
v = tf.variable(tf.constant(1.0,shape=[1]),name="v")



###############################################################
#下面代码说明如何通过tf.variable_scope函数来控制tf.get_variable函数来获取已经创建过的变量
#在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable(
        "v",[1],initializer=tf.constant_initializer(1.0)    )

#因为在命名空间foo中已经存在名字为v的变量，所以下面代码将会报错:
#Variable foo/v already exists, disallowed. Did you mean to set reuse=True in arscope
with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1])

#在生成上下文管理器时，将参数reuse设置为True，这样tf.get_variable函数将直接获取已经声明的变量
with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)  #输出为True，代表v，v1代表的是相同的Tensorflow中变量

#将参数reuse设置为True时，tf.variable_scope将只能获取已经创建过的变量，因为在命名空间bar中还没有创建变量v
#所以下面代码会报错：
#Variable bar/v dose not exists, disallowed. Did you mean to set reuse=None in Varscope?
with tf.variable_scope("bar", reuse = True):
    v = tf.get_variable("v",[1])



########################################
#下面的程序说明了当tf.variable_scope函数嵌套时，reuse参数的取值是如何确定的
with tf.variable_scope("root"):
    #可以通过tf.get_variable_scope().reuse函数来获取当前上下文管理器中reuse参数的取值
    print(tf.get_variable_scope().use)    #输出False，即最外层reuse是False

    with tf.variable_scope("foo",reuse=True):       #新建一个嵌套的上下文管理器，
                                                    #并指定reuse为True
        print(tf.get_variable_scope().reuse)        #输出True
        with tf.variable_scope("bar"):              #新建一个嵌套的上下文管理器但不指定
                                                    #reuse，这时reuse的取值会和外面一层保持一致
            print(tf.get_variable_scope().reuse)    #输出True
    print(tf.get_variable_scope().reuse)            #输出False。退出reuse设置为True的上下文之后
                                                    #reuse的值又回到了False


#############################################
#以下代码显示了如何通过tf.variable_scope来管理变量的名称
v1 = tf.get_variable("v",[1])
print(v1.name)      #输出v:0，“v”为变量的名称，“：0”表示这个变量是生成变量这个运算的第一个结果

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v",[1])
    print(v2.name)      #输出foo/v:0。在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称
                        #并通过/来分隔命名空间的名称和变量的名称

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[1])
        print(v3.name)  #输出foo/bar/v:0。命名空间可以嵌套，同时变量的名称也会加入所有命名空间的名称作为前缀

    v4 = tf.get_variable("v1",[1])
    print(v4.name)      #输出foo/v1:0。当命名空间退出之后，变量名称也就不会再被加入其前缀了

#创建一个名称为空的命名空间，并设置reuse=True
with tf.variable_scope("",reuse=true):
    v5 = tf.get_variable("foo/bar/v",[1])       #可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量
                                                #比如这里通过指定名称foo/bar/v来获取在命名空间foo/bar/中创建的变量
    print(v5==v3)                               #输出True
    v6 = tf.get_variable("foo/v1",[1])
    print(v6 == v4)                             #输出True


##############################################
#以下代码对5.2.1小节中定义的计算前向传播结果的函数做了一些改进
def inference(input_tensor, reuse=False):
    #定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope('layer1',reuse=reuse):
        #根据传进来的reuse来判断是创建新变量还是使用已经创建好的。在第一次构造网络时需要创建新变量
        #以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传进来了
        weights = tf.get_variable("weights",[INPUT_NODE, LAYER1_NODE],
                                  initializer = tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",[OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+ biases
    #返回最后的前向传播结果
    return layer2

x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
y = inference(x)

#在程序中需要使用训练好的神经网络进行推导时，可以直接调用inference(new_x, True)
#如果需要使用滑动平均模型可以参考5.2.1小节中使用的代码，把计算滑动平均的类传到
#inference函数中即可，获取或者创建变量的部分不需要改变
new_x = ...
new_y = inference(new_x,true)
###############################################
#使用上面这段代码所示的方式，就不再需要将所有变量都作为参数传递到不同的函数中了。当神经网络结构更加复杂、参数更多时，
#使用这种变量管理的方式将大大提高程序的可读性


