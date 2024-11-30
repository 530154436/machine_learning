### 学习率调度器
#### 学习率预热(warmup)
warmup是针对学习率learning rate优化的一种策略。 <br>
主要过程是：在预热期间，学习率从0线性（也可非线性）增加到优化器中的初始预设lr，之后使其学习率从优化器中的初始lr线性降低到0。<br>
学习率变化：上升 -> 平稳 -> 下降 <br>

`为什么用warmup？`(warmup的作用)
+ 稳定性：开始训练时，模型的权重(weights)是随机初始化的，此时若选择一个较大的学习率,可能带来模型的不稳定(振荡)。通过预热学习率，可以避免初始阶段模型的不稳定性，防止模型在训练开始时发散或无法收敛。
+ 收敛加速：预热阶段使用较低的学习率可以帮助模型更快地收敛，加快训练速度。
+ 泛化性能：适当的学习率预热可改善模型的泛化能力，使其更好地适应训练数据的分布。
+ 探索性：较低的学习率在初始阶段有助于模型在参数空间更广泛地探索，有助于找到全局最优点或局部最优点。

`适用场景`：在预训练模型（如BERT、GPT等）的基础上进行微调
+ 预训练模型一般具有大量的参数，而微调时的数据集相对较小。
+ 使用预训练模型的学习率调度器，可以在训练初始阶段进行学习率的预热，使得模型可以更稳定地收敛，避免在初始阶段学习率过大导致模型性能下降。


```python
from transformers import get_linear_schedule_with_warmup
```
+ optimizer ([`~torch.optim.Optimizer`]): 优化器
+ num_warmup_steps (`int`): 初始预热步数 = int(len(train_loader) * n_epochs * 0.01)
+ num_training_steps (`int`): 整个训练过程的总步数 = len(train_loader) * n_epochs
+ last_epoch (`int`, *optional*, defaults to -1):

注意：当num_warmup_steps参数设置为0时，learning rate没有预热的上升过程，只有从初始设定的learning rate 逐渐衰减到0的过程


[Transformers自定义学习了动态调整](https://www.ylkz.life/deeplearning/p10462014/)
[学习率预热(transformers.get_linear_schedule_with_warmup)](https://blog.csdn.net/orangerfun/article/details/120400247)
[模型层间差分学习率](https://www.cnblogs.com/gongyanzh/p/16127167.html)


### Label Studio
> 官网：https://labelstud.io
> pip install label-studio==1.8.2

C:\Users\chubin.zheng\AppData\Local\label-studio\label-studio

conda activate wzalgo_recommender_nlp_gpu
label-studio

+ 导入文件
[LabelStudio待标注数据加入模型预测数据](https://labelstud.io/guide/predictions)

+ 导出文件格式转换
[LabelStudio标注数据 =转换=> conll格式](https://github.com/HumanSignal/label-studio-converter/blob/master/tests/test_export_conll.py)


### DeepKe
[浙江大学信息抽取](https://github.com/zjunlp/DeepKE/tree/2.2.6)


[【NLP】基于BERT-BiLSTM-CRF的NER实现](https://zhuanlan.zhihu.com/p/518834713)

[Pytorch训练代码框架](https://zhuanlan.zhihu.com/p/484937009)
[Pytorch训练代码框架-GitHub](https://github.com/ifwind/code_framework_pytorch/)


如何高效管理深度学习实验？
https://zhuanlan.zhihu.com/p/379464474
https://github.com/L1aoXingyu/Deep-Learning-Project-Template
