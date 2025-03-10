
1. 《PyTorch深度学习实践》 <br>
[《PyTorch深度学习实践》课程](https://liuii.github.io/post/pytorch-tutorials/)<br>
[《PyTorch深度学习实践》笔记](https://github.com/MLNLP-World/Pytorch-Deep-Learning-Practice-Notes/tree/main)<br>
[《PyTorch深度学习实践》代码](https://github.com/DelinQu/pytorch-prev/tree/master)<br>

2. 《动手深度学习v2.0》 <br>
[在线课程](https://courses.d2l.ai/zh-v2/)<br>
[视频地址](https://www.bilibili.com/video/BV1if4y147hS/?spm_id_from=333.999.0.0)<br>
[在线书籍](https://zh.d2l.ai/index.html)<br>
[项目代码](https://github.com/d2l-ai/d2l-zh)<br>

+ 系统版本：lsb_release -a
```
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy
```

+ 显卡版本(虚拟机)：nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-PCI...  Off  | 00000000:36:00.0 Off |                    0 |
| N/A   37C    P0    42W / 250W |      0MiB / 20268MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

conda create -n deeplearning python==3.10.12
conda activate deeplearning

python -m pip install --no-cache -r 00-tools/requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host mirrors.tuna.tsinghua.edu.cn
