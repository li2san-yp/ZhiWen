import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import reuters
import keras
from typing import cast
import numpy as np
from keras import layers, optimizers, losses, metrics, activations
from keras.src.callbacks.history import History
from matplotlib import pyplot as plt
"""
    单标签，多分类问题
    这个代码将标签编码改为了整数编码，jupyter中的代码是单热点编码，区别在于：
        使用的损失函数和metrics要加上sparse前缀；
        后面history的键名加上sparse前缀
"""

# 向量化输入数据的函数
def vectorize_sequences(sequences, dimension = 10000):
    rst = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        rst[i, sequence] = 1
    return rst

# 导入数据集
(x_train, y_train),(x_test, y_test) = reuters.load_data(num_words=10000)

# 处理数据集
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

# 定义链接层
model = keras.models.Sequential()
model.add(keras.Input((10000,)))
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(46, activation=activations.softmax))

# 使用 优化器：小批量随机梯度下降 损失函数：分类交叉熵 跟踪准确率
model.compile(
    optimizer="rmsprop",
    loss=losses.sparse_categorical_crossentropy,
    metrics=[metrics.sparse_categorical_accuracy]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=20,
    validation_data=(x_test, y_test)
)

history = cast(History, history)


# 数据可视化
dic = history.history
print(dic.keys())
categorical_acc = dic["sparse_categorical_accuracy"]
loss = dic["loss"]
val_categorical_acc = dic["val_sparse_categorical_accuracy"]
val_loss = dic["val_loss"]
epochs = range(1,len(categorical_acc)+1)

plt.plot(epochs, categorical_acc, "bo", label = "categorical_acc")
plt.plot(epochs, val_categorical_acc, "b", label = "val_categorical_acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.show()

plt.clf()
plt.plot(epochs, loss, "bo", label = "loss")
plt.plot(epochs, val_loss, "b", label = "val_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.xticks(epochs)
plt.show()
