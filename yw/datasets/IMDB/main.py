import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import imdb
from typing import cast
import numpy as np
import keras
from keras import models, layers
from keras.src.callbacks.history import History
import matplotlib.pyplot as plt

"""
    这个代码基于imdb数据集实现一个电影评论的二分类问题
"""

# 辅助函数，用来将输入的数据向量化，由于只取频率最高的前10000个词，所以列大小为10000，如果出现则对应列标为1
def vectorize_sequences(sequences,dimension = 10000):
    rst = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        rst[i,sequence] = 1
    return rst

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 数据加载之后，data是二维数组，行表示样本，列表示每个样本中包含的单词索引，这里只取前10000个单词，后面的单词认为频率不高，直接舍弃
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels, dtype=np.float32) # 标签只有一个，即该评论是正向还是负向
y_test = np.asarray(test_labels, dtype=np.float32)

# 定义各个层，第一个层是规定输入的大小，每个样本被规范化为一个长10000的向量，所以输入为10000维
# 另外加2个线性层，使用激活函数relu
# 最后使用sigmoid函数归一化数据
model = models.Sequential()
model.add(layers.Input((10000,)))
model.add(layers.Dense(16,activation=keras.activations.relu)) # 添加层时，需要激活函数，relu是常用的激活函数
model.add(layers.Dense(16,activation=keras.activations.relu))
model.add(layers.Dense(1,activation=keras.activations.sigmoid))

x_val = x_train[10000:]
partial_x_train = x_train[:10000]
y_val = y_test[10000:]
partial_y_train = y_train[:10000]

model.compile(
    optimizer="rmsprop",
    loss=keras.losses.binary_crossentropy,
    metrics=["accuracy"]
)

history = model.fit(
    partial_x_train,
    partial_y_train,
    batch_size=512,
    epochs=4,
    validation_data=(x_val,y_val)
)
history = cast(History, history)

his_dic = history.history

loss_values = his_dic["loss"]
val_loss_values = his_dic["val_loss"]
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,"bo",label = "training loss")
plt.plot(epochs,val_loss_values,"b",label="validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.clf()
acc = his_dic["accuracy"]
val_acc = his_dic["val_accuracy"]
plt.plot(epochs, acc, "bo", label = "training acc")
plt.plot(epochs, val_acc, "b", label = "validation acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
