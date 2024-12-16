
## 一、PyTorch 模型模块

将拟合模型的任务分解为两个关键问题：
+ 优化（optimization）：用模型拟合观测数据的过程
+ 泛化（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

一般来说，深度学习的主要步骤是：
+ 搭建计算图 （相当于输入自己的公式，一般来说是线性公式 $y=\omega x+b$）
+ 前向传播 （相当于计算公式的值，即 $y$）
+ 计算损失 （相当于计算 $y$ 与真实值的差距，即 $loss$）
+ 反向传播 （相当于计算 $loss$ 对 $\omega$ 和 $b$ 的偏导数，即 $\frac{\partial loss}{\partial \omega}$ 和 $\frac{\partial loss}{\partial b}$）
+ 更新参数 （相当于更新 $\omega$ 和 $b$，即 $\omega=\omega-\alpha \frac{\partial loss}{\partial \omega}$ 和 $b=b-\alpha \frac{\partial loss}{\partial b}$）


## 参考引用

[1] [《PyTorch实用教程》（第二版）](https://github.com/TingsongYu/PyTorch-Tutorial-2nd/releases/tag/v1.0.0)<br>
[2] [《深入浅出PyTorch》](https://github.com/datawhalechina/thorough-pytorch)<br>
[3] [PyTorch中文文档](https://www.bookstack.cn/read/PyTorch-cn/README.md)<br>
