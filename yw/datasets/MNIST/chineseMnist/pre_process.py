import os
import matplotlib.pyplot as plt
import numpy as np
import json
dataset_path = "G:/多模态知识问答系统/数据集/chineseMNIST/data"

data = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        content = plt.imread(f"{root}/{file}")
        content = content.reshape((64*64,))
        data.append({
            "path":root+"/"+file,
            "label":file.rsplit("_",1)[-1].split(".",1)[0]
        })
        print(f"图片：{root}/{file} 处理完成！")
    
    with open(f"./chineseMNIST.json",mode="w",encoding="utf-8") as file:
        file.write(json.dumps(data))
    break

"""
    生成的json文件格式大致如下：
[
    {
        "path": "G:/\u591a\u6a21\u6001\u77e5\u8bc6\u95ee\u7b54\u7cfb\u7edf/\u6570\u636e\u96c6/chineseMNIST/data/input_100_10_1.jpg",
        "label": "1"
    },
    {
        "path": "G:/\u591a\u6a21\u6001\u77e5\u8bc6\u95ee\u7b54\u7cfb\u7edf/\u6570\u636e\u96c6/chineseMNIST/data/input_100_10_10.jpg",
        "label": "10"
    },
    {
        "path": "G:/\u591a\u6a21\u6001\u77e5\u8bc6\u95ee\u7b54\u7cfb\u7edf/\u6570\u636e\u96c6/chineseMNIST/data/input_100_10_11.jpg",
        "label": "11"
    }
    ...
]
"""
