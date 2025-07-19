import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import matplotlib.pyplot as plt
import keras
from keras import models, layers, activations, losses, optimizers, metrics
from keras.utils import to_categorical
class ChineseMNIST:
    def __init__(self):
        # 导入数据
        self.json_data = json.loads(open("./chineseMNIST.json",mode="r",encoding="utf-8").read())
        np.random.shuffle(self.json_data)

    def load_data(self,num_train:int = 10000) -> tuple[tuple[np.ndarray, np.ndarray],tuple[np.ndarray, np.ndarray]]:
        """
        args:
            num_train: size of train dataset. 
            range: (0, 15000) 
            the num_test will be 15000 - num_train.
        return:
            (x_train, y_train), (x_test, y_test)
        """
        if not (num_train > 0 and num_train < 15000):
            raise ValueError(
                "num_train must in the range (0,15000)"
            )
        self.x_train = np.zeros((num_train,64*64),dtype=np.int8)
        self.y_train = np.zeros((num_train),dtype=np.int64)
        self.x_test = np.zeros((15000-num_train,64*64),dtype=np.int8)
        self.y_test = np.zeros((15000-num_train),dtype = np.int64)
        
        for i, sample in zip(range(num_train), self.json_data[:num_train]):
            path = sample["path"]
            label = int(sample["label"]) - 1
            self.x_train[i, range(64*64)] = plt.imread(path).flatten() # 注意这里要把导入的图片展平为一维向量
            self.y_train[i] = label
        for i, sample in zip(range(num_train,15000), self.json_data[num_train:]):
            path = sample["path"]
            label = int(sample["label"]) - 1
            self.x_test[i-num_train, range(64*64)] = plt.imread(path).flatten() # 这里也要展平
            self.y_test[i-num_train] = label
        return (self.x_train, self.y_train),(self.x_test, self.y_test)


if __name__ == "__main__":
    dat = ChineseMNIST()
    (x_train, y_train), (x_test, y_test) = dat.load_data()
    y_train = to_categorical(y_train, 15)
    y_test = to_categorical(y_test, 15)
    
    # 使用卷积神经网络，预测准确率很高
    # x_train = x_train.reshape((-1,64,64,1))
    # x_test = x_test.reshape((-1,64,64,1))
    # model = models.Sequential([
    #     layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    #     layers.MaxPooling2D((2,2)),
    #     layers.Conv2D(64, (3,3), activation='relu'),
    #     layers.MaxPooling2D((2,2)),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(15, activation='softmax')
    # ])
    
    model = models.Sequential()
    model.add(layers.Input((4096,)))
    model.add(layers.Dense(64, activations.relu))
    model.add(layers.Dense(64, activations.relu))
    model.add(layers.Dense(15, activations.softmax))
    
    model.compile(
        optimizer="rmsprop",
        loss=losses.categorical_crossentropy,
        metrics=[metrics.categorical_accuracy]
    )
    
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=20,
        validation_data=(x_test, y_test)
    )

    dic = history.history
    acc = dic["categorical_accuracy"]
    val_acc = dic["val_categorical_accuracy"]
    loss = dic["loss"]
    val_loss = dic["val_loss"]
    epochs = range(1,len(acc)+1)
    plt.plot(epochs, acc, "bo", label = "acc")
    plt.plot(epochs, val_acc, "b", label = "val_acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    plt.xticks(epochs)
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
