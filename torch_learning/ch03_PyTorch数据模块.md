
## 一、PyTorch 数据模块
PyTorch 通常使用 Dataset 和 DataLoader 这两个工具类来构建数据管道。
Dataset 定义了数据集的内容，类似于列表或字典等数据结构，具有确定的长度，支持通过索引访问数据集中的元素，从而实现索引到样本的映射。
DataLoader 负责高效地按批次加载数据，支持数据预处理和并行加载。DataLoader 包含两个核心组件：Sampler 和 Dataset。
Sampler 决定如何生成数据的索引顺序（如随机采样、顺序采样等），而 Dataset 根据 Sampler 提供的索引从数据集中提取对应的样本。
```mermaid
graph LR
    A[Dataset] --> B[DataLoader]
    B[DataLoader] --Sampler--> C[Batch]
    C[Batch] --> D[Model]
```

### 1.1 Dataset
PyTorch提供了两种数据集定义方式：映射式数据集、可迭代数据集。

#### 1.1.1 映射风格数据集（Map-style datasets）
映射式数据集是实现了 `__getitem__()` 和 `__len__()` 协议的数据集，它表示从索引（可能是非整数） 或键到数据样本的映射。
在加载数据时，PyTorch将自动使用迭代索引(如enumerate)作为key，通过字典索引的方式获取value，本质就是将数据集定义为一个字典。
```python
from torch.utils.data import Dataset
class MyMapDataset(Dataset):
    def __init__(self):
        self.data = {
            "sample1": "data1",
            "sample2": "data2",
            "sample3": "data3"
        }
    def __getitem__(self, key):
        return self.data[key]  # 通过字符串键访问数据样本
    def __len__(self):
        return len(self.data)  # 数据集的大小

# 创建数据集实例
dataset = MyDataset()

# 访问数据
print(dataset["sample1"])  # 输出 "data1"
print(len(dataset))  # 输出 3
```
在 整数索引 的示例中，数据集通过 dataset[index] 访问数据。
在 键映射 的示例中，数据集通过 dataset[key] 访问数据，其中 key 是字符串类型的标识符。
映射式数据集的特点就是可以通过索引或键来访问数据，而不仅限于按位置索引，从而在不同的应用场景中更加灵活。

#### 1.1.2 可迭代数据集（Iterable-style datasets）


### 1.2 DataLoader
### 1.3 常用API
### 1.4 常用数据集

### 参考引用

[1] [20天吃掉那只Pytorch-5-1, Dataset和DataLoader](https://jackiexiao.github.io/eat_pytorch_in_20_days/5.%E4%B8%AD%E9%98%B6API/5-1%2CDataset%E5%92%8CDataLoader/)<br>
[2] [《深入浅出PyTorch》](https://github.com/datawhalechina/thorough-pytorch)<br>
[3] [Pytorch建模过程中的DataLoader与Dataset](https://www.cnblogs.com/chenhuabin/p/17026018.html)<br>
[4] [PyTorch官方文档-data](https://pytorch.org/docs/stable/data.html)<br>
[5] [PyTorch中文文档-data](https://pytorch.ac.cn/docs/2.4/data.html)<br>
[6] [《PyTorch实用教程》（第二版）](https://github.com/TingsongYu/PyTorch-Tutorial-2nd/releases/tag/v1.0.0)<br>

