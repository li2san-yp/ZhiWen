import os
import shutil

"""
    该代码从原始的数据集拿猫狗图片各2000张并拆分为训练集，验证集和测试集
"""

base_path = "G:\\多模态知识问答系统\\数据集\\dogs-vs-cats\\train"
target_path = "G:\\多模态知识问答系统\\数据集\\dogs_and_cat_dataset"

def makedir(path: str):
    if not os.path.exists(path):
        print(f"创建文件夹成功：{path}")
        os.mkdir(path)

def copypic(src, dst, start, end, animal) -> None:
    """
    Args:
        src: the source path.
        dst: the destination path.
        start and end: copy pic range: [start, end)
        animal: cat or dog
    Returns:
        None
    """
    if animal == "cat":
        fnames = [f"cat.{i}.jpg" for i in range(start, end)]
        for name in fnames:
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if not os.path.exists(d):
                shutil.copyfile(s, d)
    elif animal == "dog":
        fnames = [f"dog.{i}.jpg" for i in range(start, end)]
        for name in fnames:
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if not os.path.exists(d):
                shutil.copyfile(s, d)
    else:
        raise ValueError(
            "the argument \"animal\" must be one of [cat, dog]."
        )


makedir(target_path)
train_dir = os.path.join(target_path, "train")
validation_dir = os.path.join(target_path, "validation")
test_dir = os.path.join(target_path, "test")
makedir(train_dir)
makedir(validation_dir)
makedir(test_dir)

train_cat_dir = os.path.join(train_dir, "cat")
train_dog_dir = os.path.join(train_dir, "dog")
validation_cat_dir = os.path.join(validation_dir, "cat")
validation_dog_dir = os.path.join(validation_dir, "dog")
test_cat_dir = os.path.join(test_dir, "cat")
test_dog_dir = os.path.join(test_dir, "dog")

makedir(train_cat_dir)
makedir(train_dog_dir)
makedir(validation_cat_dir)
makedir(validation_dog_dir)
makedir(test_cat_dir)
makedir(test_dog_dir)

copypic(base_path, train_cat_dir, 0, 1000, "cat")
copypic(base_path, train_dog_dir, 0, 1000, "dog")
copypic(base_path, validation_cat_dir, 1000, 1500, "cat")
copypic(base_path, validation_dog_dir, 1000, 1500, "dog")
copypic(base_path, test_cat_dir, 1500, 2000, "cat")
copypic(base_path, test_dog_dir, 1500, 2000, "dog")

print(len(os.listdir(train_cat_dir)))
