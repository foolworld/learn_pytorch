# Dataset类整体使用流程

## 1. **创建对象**

python

```
dataset = MyDataset(root_dir="训练数据路径", laber_dir="标签文件夹名")
```



**作用**：告诉Dataset图片在哪里

## 2. **检查数据量**

python

```
num = len(dataset)  # 返回这个文件夹里有多少张图片
```



## 3. **获取单张图片**

python

```
img = dataset[下标]  
# 例如：dataset[0] 获取第一张
# dataset[5] 获取第六张
```



## 4. **具体执行步骤（按代码逻辑）**

### 4.1 初始化时发生的事情

python

```
def __init__(self, root_dir, laber_dir):
    # 1. 保存路径
    self.root_dir = "训练数据路径"  # 你传的第一个参数
    self.laber_dir = "标签文件夹名" # 你传的第二个参数
    
    # 2. 拼接路径
    self.path = os.path.join("训练数据路径", "标签文件夹名")
    # 结果示例："data/train/ants"
    
    # 3. 获取所有图片文件名（你代码有错但应该是这个意思）
    # self.image_path = ["ants1.jpg", "ants2.jpg", ...]
```



### 4.2 获取图片时发生的事情

python

```
def __getitem__(self, index):
    # 1. 通过索引找文件名
    img_name = self.image_path[index]  
    # index=0 → img_name = "ants1.jpg"
    
    # 2. 拼接完整路径（这里代码有bug）
    img_item_path = os.path.join(self.root_dir, img_name)
    # 应该是：os.path.join(self.path, img_name)
    # 结果："data/train/ants/ants1.jpg"
    
    # 3. 打开图片
    img = Image.open(img_item_path)  # 得到PIL图片对象
```



### 4.3 获取数量时发生的事情

python

```
def __len__(self):
    # 直接返回图片文件名列表的长度
    return len(self.image_path)  # 比如有100张就返回100
```



## 5. **子函数总结**

### os.path.join()

python

```
# 功能：拼接路径
path = os.path.join("folder1", "folder2", "file.jpg")
# 结果：folder1/folder2/file.jpg（Windows会自动转\）
```



### os.listdir()

python

```
# 功能：获取文件夹内所有文件名
files = os.listdir("文件夹路径")
# 结果：["file1.jpg", "file2.jpg", "subfolder", ...]
```



**注意**：你代码里缺少这个函数，但必须要有

### Image.open()

python

```
# 功能：打开图片文件
img = Image.open("图片路径")
# 结果：PIL Image对象，可以显示、保存、转换
```



## 6. **最终正确使用流程**

python

```
from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_list = os.listdir(self.path)  # 关键：获取文件名列表
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.path, img_name)  # 正确拼接
        img = Image.open(img_path)
        return img, self.label_dir  # 返回图片和标签
    
    def __len__(self):
        return len(self.img_list)

# 使用
dataset = MyDataset("hymenoptera_data/train", "ants")
print(f"总数: {len(dataset)}")
img, label = dataset[0]  # 获取第一张图片和标签"ants"
```