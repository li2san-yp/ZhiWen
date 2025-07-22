from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import cast
import os

"""
    这段代码实现了一个简单的图像数据迭代器，原本的目的是想了解一下keras的数据加载器，但是后面写代码才发现keras早就弃用了。
    这个代码无关紧要，后面会尝试使用pytorch实现main的代码，可以拿来理解pytorch中的DataLoader的工作原理
"""
# 本函数接收图片路径，并返回图片转成的张量
def image_to_tensor(image_path):
    """
    Args:
        image_path: path to the picture.
    Returns:
        the "np.ndarray" of the picture by the mode of "RGB", shape: (150, 150, 3) 
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img_resized = img.resize((150, 150))
    
    img_tensor = np.array(img_resized, dtype=np.uint8)
    return img_tensor

# 图片迭代器，每次迭代返回20张图片
class ImgGenerator:
    def __init__(self, path: str | list[str], labels, batch_size = 20):
        """
        Args:
            path: 
                the dataset path.    it supports the type of "str" or "list[str]"
            labels: 
                the label of every sample, it should have the same size of len(filenames).
                (filenames -- os.listdir(path)    if have two or more paths, take sum of it.)
            batch_size:
                the batch of every iter.
        """
        self.path = path
        self.batch_size = batch_size
        self.labels = labels
        self.filenames = []
        self.image_size = None
        if isinstance(self.path, str): # 传入path只有一个，遍历特征标签即可
            self.filenames = os.listdir(path) # type: ignore
            if len(self.filenames) != len(self.labels):
                raise ValueError(
                    "size of len(filenames) and len(labels) must be equal.\nPLease check your labels."
                )
            self.filenames = [(self.path+"\\"+filename, label) for filename, label in zip(self.filenames, self.labels)]
        elif isinstance(self.path, list): # 传入path有多个，先生成filename列表，再加入标签
            for p in self.path:
                for filename in os.listdir(p):
                    self.filenames.append(p+"\\"+filename)
            if len(self.filenames) != len(self.labels):
                raise ValueError(
                    "size of len(filenames) and len(labels) must be equal.\nPLease check your labels."
                )
            self.filenames = [(filename,label) for filename, label in zip(self.filenames, self.labels)]
        else:
            raise TypeError(
                "the type of argument \"path\" must be str or list[str]"
            )
                    
        np.random.shuffle(self.filenames) # 打乱顺序
        self.index = 0
        self.max_index = len(self.filenames)

    def __iter__(self):
        return self
    def __next__(self):
        if self.max_index <= self.index: # 超出索引范围
            raise StopIteration
        elif self.max_index - self.index <= self.batch_size: # 未超出索引范围，但是剩余数据小于batch_size
            rest = self.max_index - self.index
            file_list = np.zeros((rest, 150, 150, 3))
            label_list = np.zeros((rest,))
            for i in range(rest):
                file_list[i] = image_to_tensor(self.filenames[self.index+i][0])
                label_list[i] = self.filenames[self.index+i][1]
            self.index += rest
            return file_list, label_list
        else: # 未超出范围且剩余数据充足
            file_list = np.zeros((self.batch_size, 150, 150, 3))
            label_list = np.zeros((self.batch_size,))
            for i in range(self.batch_size):
                file_list[i] = image_to_tensor(self.filenames[self.index+i][0])
                label_list[i] = self.filenames[self.index+i][1]
            self.index += self.batch_size
            return file_list, label_list

# 下面的代码使用迭代器获取20张图片并显示
# labels = np.concatenate([[np.zeros((1000,))],[np.ones((1000,))]],axis=1).reshape((2000,))
# imggen = ImgGenerator(["G:\\datasets\\dogs_and_cat_dataset\\train\\cat", "G:\\datasets\\dogs_and_cat_dataset\\train\\dog"], labels)

# batch = next(imggen)

# _, axes = plt.subplots(4, 5, figsize=(12, 8))
# axes = cast(np.ndarray, axes)
# axes = axes.flatten()

# images, labels = batch
# for ax, sample, label in zip(axes, images, labels):
#     ax = cast(Axes, ax)
#     sample = cast(np.ndarray, sample)
#     ax.axis("off")
#     ax.set_title("dog" if label == 1 else "cat")
#     ax.imshow(sample.astype(np.uint8))

# plt.tight_layout()
# plt.show()
