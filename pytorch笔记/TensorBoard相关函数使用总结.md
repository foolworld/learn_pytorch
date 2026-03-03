# TensorBoard相关函数使用总结

## 1. **整体执行流程**

text

```
1. 创建SummaryWriter → 2. 准备数据 → 3. 添加数据到TensorBoard → 4. 关闭writer
```



## 2. **每个函数的使用方法**

### 2.1 `SummaryWriter("logs")`

python

```
writer = SummaryWriter("logs")
```



**作用**：创建TensorBoard日志写入器
**参数**：`"logs"` - 日志保存的文件夹路径
**效果**：在当前目录创建`logs/`文件夹，保存所有可视化数据

### 2.2 `Image.open(image_path)`

python

```
image_PIL = Image.open(image_path)
```



**作用**：用PIL库打开图片文件
**参数**：`image_path` - 图片文件的路径字符串
**返回**：PIL Image对象

### 2.3 `np.array(image_PIL)`

python

```
img_array = np.array(image_PIL)
```



**作用**：将PIL Image对象转换为numpy数组
**参数**：`image_PIL` - PIL Image对象
**返回**：numpy数组，形状为(H, W, C)

- H: 高度（像素行数）
- W: 宽度（像素列数）
- C: 通道数（RGB为3，灰度图为1）

### 2.4 `writer.add_image()`

python

```
writer.add_image("test", img_array, 1, dataformats="HWC")
```



**作用**：将图片添加到TensorBoard
**参数**：

- `"test"`: 标签名，在TensorBoard中显示的名称
- `img_array`: 图片数据，必须是numpy数组
- `1`: 步骤数（step），可以理解为图片序号或训练步数
- `dataformats="HWC"`: 指定数组格式
  - `HWC`: 高度×宽度×通道（Height, Width, Channel）
  - 也可以是`CHW`（通道×高度×宽度）

### 2.5 `writer.add_scalar()`

python

```
writer.add_scalar("y=2x", 3*i, i)
```



**作用**：将标量数据（数字）添加到TensorBoard
**参数**：

- `"y=2x"`: 标签名，会显示为图表标题
- `3*i`: y轴数值（数据点值）
- `i`: x轴数值（步骤数）

**循环中的效果**：

- 当`i=0`时：添加点(0, 0)
- 当`i=1`时：添加点(1, 3)
- 当`i=2`时：添加点(2, 6)
- ...
- 当`i=99`时：添加点(99, 297)

### 2.6 `writer.close()`

python

```
writer.close()
```



**作用**：关闭SummaryWriter，确保所有数据写入文件

## 3. **数据类型转换流程**

text

```
图片文件(.jpg) 
    ↓ Image.open()
PIL Image对象 
    ↓ np.array() 
numpy数组(H,W,C) 
    ↓ writer.add_image()
TensorBoard可视化
```



## 4. **实际使用示例**

python

```
# 1. 创建写入器（开始记录）
writer = SummaryWriter("实验记录")

# 2. 记录一个数字变化（比如训练loss）
for epoch in range(100):
    loss = 1.0 / (epoch + 1)  # 模拟loss下降
    writer.add_scalar("训练损失", loss, epoch)

# 3. 记录一张图片
img = np.random.rand(256, 256, 3)  # 创建随机图片
writer.add_image("随机图片", img, 0)

# 4. 结束记录
writer.close()
```



## 5. **查看结果**

bash

```
# 在命令行启动TensorBoard
tensorboard --logdir=logs

# 浏览器访问
http://localhost:6006
```