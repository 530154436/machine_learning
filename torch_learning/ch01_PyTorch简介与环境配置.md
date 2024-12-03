

### 一、PyTorch 初认识

#### 1.1 PyTorch历史
PyTorch 是 Facebook AI Research (FAIR) 于 2017 年发布的深度学习框架，其名称结合了 Python 和早期的深度学习框架 Torch。
Torch 是纽约大学在 2002 年开发的框架，最初采用小众的 Lua 语言作为接口，使用门槛较高，因此团队后续用 Python 语言对其进行了重构，最终诞生了 PyTorch。

PyTorch 的关键发展历程包括：
+ 2016 年 8 月，发布第一个公开版本 v0.1.1。
+ 2017 年 1 月，正式发布 PyTorch。
+ 2018 年 4 月，更新 0.4.0 版，支持 Windows 系统。
+ 2018 年 11 月，发布 1.0 稳定版，成为 GitHub 上增长第二快的开源项目。
+ ...

最新版本于 2024 年 10 月发布PyTorch 2.5.1。[PyTorch-releases](https://github.com/pytorch/pytorch/releases)


#### 1.2 PyTorch发展趋势

PyTorch近年来发展迅猛，已成为深度学习框架中的佼佼者，尤其在学术界大放异彩。绝大多数顶会论文都选择PyTorch实现，其在学术界的使用比例逐年上升，并在2018-2019年间实现了对TensorFlow的超越。 
目前，PyTorch的学术占有率甚至达到了TensorFlow的两倍以上，稳坐学术界“带头大哥”的位置。<br>

此外，PyTorch在工业界的表现也逐步赶超。
早期工业部署方面，PyTorch稍逊于TensorFlow，但随着libtorch、TorchServe等工具的推出，以及适配性良好的部署框架（如TensorRT、OpenVINO、ONNX等）的支持，PyTorch在部署效率和灵活性上得到了显著提升。<br>

PyTorch的优势在于快速实现从研究原型到生产部署的转化。
无论是学术研究者追求最新模型，还是工业开发者需要应用新技术，PyTorch都成为了首选框架。<br>

<img src="images/PyTorch发展趋势.png" width="50%" height="30%" alt=""><br>


#### 1.3 环境配置

+ 使用镜像
```shell
nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
```

+ 系统版本：lsb_release -a
```
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy
```

+ 显卡版本(虚拟机)：nvidia-smi
```
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
```

+ Miniconda创建虚拟环境
```shell
conda create -n deeplearning python==3.10.12
conda activate deeplearning
python -m pip install --no-cache  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --trusted-host mirrors.tuna.tsinghua.edu.cn
```


[1] [《PyTorch实用教程》（第二版）](https://github.com/TingsongYu/PyTorch-Tutorial-2nd/releases/tag/v1.0.0)<br>
[2] [PyTorch 2.4.0](https://pytorch.org/get-started/previous-versions/#v240)<br>
[3] [PyTorch中文文档](https://www.bookstack.cn/read/PyTorch-cn/README.md)<br>
