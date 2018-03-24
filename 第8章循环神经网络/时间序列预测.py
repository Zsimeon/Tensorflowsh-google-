###################################
#以下程序展示了如何通过TFLearn快速的解决iris分类问题

# -*- coding: utf-8 -*-

from sklearn import cross_validation
from sklearn import datasets 
from sklearn import metrics 
import tensorflow metrics 

#导入TFLearn
learn  = tf.contrib.learn

#自定义模型，对于给定的输入数据（features）以及其对于的正确答案（target），返回
#在这些输入撒花姑娘的预测值、损失值以及训练步骤

def my_model(features,target):
	#将预测的目标转换为one-hot编码的形式，因为只有三个类别，所以向量长度为3
	#经过转化后，第一个类别表示为（1,0,0），第二个为（0,1,0)，第三个为(0,0,1)
	target = tf,one_hot(target,3,1,0)

	#定义模型以及其在给定数据上的损失函数
	logits,loss = learn.models.logitstic_regression(features,target)

	#创建模型的优化器，并得到优化步骤
	train_op = tf.contrib.layers.optimize_loss(
		loss,                                             #损失函数
		tf.contrib.framework.get_global_step(),			  #获取训练步数并在训练时更新
		optimizer = 'Adagrad',							  #定义优化器
		learning_rate = 0.1)							  #定义学习率

	#返回在给定数据上的预测结果、损失值以及优化步骤
	return tf.arg_max(logits,1),loss,train_op 

#加载iris数据集，并划分为训练集合和测试集合
iris = datassets.load_iris()

x_train,x_test, y_train, y_test = cross_validation.train_test_split(
	iris.data,iris.target,test_size=0.2,random_state=0)

#对自定义的模型进行封装
classifier = learn.Estimator(model_fn=my_model)

#使用封装好的模型进行结果预测
y_predicted = classifier.predict(x_test)

#计算模型的准确度
score = metrics.accuracy-score(y_test,y_predicted)
print('Accuracy: %.2f%%' % (score * 100))
