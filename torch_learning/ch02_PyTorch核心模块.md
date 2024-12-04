### 一、PyTorch 核心模块

#### 1.1 模块结构
模块代码位置：$ENS/deeplearning/Lib/site-packages/torch

| 模块                        | 功能说明                                                     |
|---------------------------|----------------------------------------------------------|
| `torch`                   | PyTorch 的主库，提供 tensor 操作、数学运算、GPU 加速等基本功能                |
| `torch.nn`                | 构建神经网络的核心模块，包含卷积、批归一化、激活、全连接、损失函数等。                      |
| `torch.nn.functional`     | 提供神经网络常用函数，如卷积、池化、激活等，功能与 `torch.nn` 中的类一致。              |
| `torch.nn.init`           | 提供神经网络参数初始化的常见策略，如常量初始化、均匀分布初始化、正态分布初始化等。                |
| `torch.optim`             | 优化器模块，提供多种优化算法（如 SGD、Adam、Adagrad 等）                     |
| `torch.autograd`          | 自动求导模块，用于自动计算模型参数的梯度                                     |
| `torch.autograd.backward` | 主要用于在求得损失函数之后进行反向梯度传播，计算梯度并更新模型参数。                       |
| `torch.autograd.grad`     | 用于计算一个标量对另一个张量的导数，并支持在计算过程中设置不参与求导的部分。                   |
| `torch.utils.data`        | 数据处理模块，提供 `DataLoader` 和 `Dataset`，简化数据加载与处理             |
| `torch.utils.tensorboard` | 用于与 TensorBoard 交互的工具，支持记录和可视化训练过程中的标量、图像、损失函数等信息        |
| `torch.cuda`              | GPU 加速模块，提供与 GPU 相关的操作                                   |
| `torch.jit`               | 即时编译器，把 Pytorch 的动态图转换成可以优化和序列化的静态图，能被 C++ 和 Java 等语言调用。 |
| `torch.onnx`              | 定义了 Pytorch 导出和加载 ONNX格式的深度学习模型描述文件，便于跨框架使用模型。           |
| `torch.multiprocessing`   | 多进程 API，可以启动不同的进程，每个进程运行不同的深度学习模型，并且能够在进程间共享张量。          |

#### 1.2 核心数据结构

torch.tensor() 只接收数据，且使用之前需要初始化，默认参数不确定，变化比较大
torch.Tensor() 接收 shape，接收数据时需要是 list[ ] 格式

### 参考引用

[1] [《PyTorch实用教程》（第二版）](https://github.com/TingsongYu/PyTorch-Tutorial-2nd/releases/tag/v1.0.0)<br>
[2] [PyTorch 2.4.0 版本发布](https://pytorch.org/get-started/previous-versions/#v240)<br>
[3] [PyTorch中文文档](https://www.bookstack.cn/read/PyTorch-cn/README.md)<br>
[4] [一览 Pytorch框架](https://zhuanlan.zhihu.com/p/334788042)<br>
[5] [PyTorch的核心模块介绍](https://blog.csdn.net/weixin_38566632/article/details/135442466)<br>

