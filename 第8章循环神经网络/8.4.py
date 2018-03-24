#2333333333333333333333333333333333
#使用ptb_raw_data函数来读取PTb的原始数据，并将原始数据中的单词转化为单词ID
#2333333333333333333333333333333333


from tensorflow.models.rnn.ptb import reader

#存放原始数据的路径
DATA_PATH = "/path/to/ptb/data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
#读取数据原始数据
print(len(train_data))

print(train_data[:100])



###########################
#截断并将数据组织成batch，Tensorflow提供了ptb_iterator函数
###########################

from tensorflow.models.rnn.ptb import reader

#类似地读取数据原始数据
DATA_PATH = "/path/to/ptb/data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

#将训练数据组织成batch大小为4，截断长度的5数据组
result = reader.ptb_iterator(train_data,4,5)
#读取第一个batch中的数据，其中包括每个时刻的输入和对应的正确输出
x,y = result.next()
print("X:",x)
print("y:",y)

