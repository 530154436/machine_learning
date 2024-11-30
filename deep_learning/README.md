

1. 《动手深度学习v2.0》 <br>
[视频地址](https://www.bilibili.com/video/BV1if4y147hS/?spm_id_from=333.999.0.0)
[在线教程](https://zh.d2l.ai/index.html)
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
