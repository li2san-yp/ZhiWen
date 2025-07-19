import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
from keras import models, layers, activations, losses, metrics
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据集处理
x_train = x_train.reshape((-1,784)) / 255
x_test = x_test.reshape((-1,784)) / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = models.Sequential()
model.add(layers.Input((784,)))
model.add(layers.Dense(64, activations.relu))
model.add(layers.Dense(64, activations.relu))
model.add(layers.Dense(10))

model.compile(
    optimizer="rmsprop",
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=20,
    validation_data=(x_test, y_test)
)

dic = history.history
acc = dic["binary_accuracy"]
val_acc = dic["val_binary_accuracy"]
loss = dic["loss"]
val_loss = dic["val_loss"]
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, "bo", label="acc")
plt.plot(epochs, val_acc, "b", label = "val_acc")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xticks(epochs)
plt.grid()
plt.show()

plt.clf()
plt.plot(epochs, loss, "bo", label="loss")
plt.plot(epochs, val_loss, "b", label = "val_loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.xticks(epochs)
plt.grid()
plt.show()
