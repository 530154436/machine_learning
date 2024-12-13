## 一、Python模块打包流程
打包Python模块是将代码打包成一个可分发的格式，以便其他人可以轻松安装和使用。通常，这涉及创建一个源分发包和一个Wheel分发包。需要安装以下依赖：
```shell
python -m pip install --upgrade build==1.2.2.post1 twine==6.0.1 setuptools==75.6.0 wheel==0.45.1
```

### 1.1 setuptools
setuptools是Python中最常用的打包和分发工具之一，它扩展了Python标准库中的distutils，提供了更强大的功能和更灵活的配置选项。setuptools的主要功能包括：

+ 依赖管理：自动管理包的依赖关系。
+ 创建和分发Wheel格式：支持创建Wheel格式的分发包，便于快速安装。 
+ 包自动发现：自动发现和包括包目录。
+ 扩展模块支持：支持构建C和C++扩展模块。
+ 入口点和控制台脚本：定义包的入口点和命令行工具。

通过setuptools，可以轻松管理Python包的构建、依赖和分发，使得包的发布和安装更加便捷和可靠。

### 1.2 wheel
wheel是Python中一种现代的打包格式，旨在取代旧的.egg格式。它是一个二进制分发格式，可以显著加快包的安装速度，因为它不需要在安装时进行编译。

+ 格式：.whl文件是一个包含预编译代码和元数据的压缩归档文件。它是PEP427中定义的标准。
+ 优势：与源代码分发包不同，Wheel包不需要在目标机器上编译，可以直接安装，大大加快了安装速度。

wheel工具通常与setuptools一起使用，并且可以通过pip进行安装。


### 1.3 创建发布包流程

+ 项目目录结构
```
torch_learning/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── torchlibrary/
│       ├── __init__.py
│       └── example.py
└── tests/
```
+ pyproject.toml
```
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "torchlibrary"
version = "0.0.0"
authors = [
  { name="zhengchubin", email="530154436@qq.com" },
]
description = "torchlibrary"
readme = "README.md"
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "torchaudio>=2.4.0",
    "jupyter>=1.1.1",
]

[project.urls]
Homepage = "https://github.com/530154436/machine_learning/tree/master/torch_learning"
Issues = "https://github.com/530154436/machine_learning/issues"
```

+ 构建发布包
```
$ python -m build

* Creating isolated environment: venv+pip...
* Installing packages in isolated environment:
  - setuptools>=61.0
  - wheel
* Getting build dependencies for sdist...
...
Successfully built torchlibrary-0.0.1.tar.gz and torchlibrary-0.0.1-py3-none-any.whl
```
## 二、发布到PyPI和测试

### 2.1 Twine
twine是一个用于将Python包上传到Python Package Index (PyPI) 及其镜像的工具。它提供了一种安全且可靠的方法来上传分发包，替代了旧的setup.py upload方法。

+ 安全性：twine使用HTTPS来上传包，并且支持使用PyPI的API tokens，避免了在命令行中暴露PyPI密码。
+ 兼容性：支持上传Wheel格式和源代码格式的分发包。
+ 简单易用：提供简单的命令行接口，可以轻松上传包。
+ 错误检测：在上传之前检查包的完整性和格式问题

通过使用twine，开发者可以安全且高效地将Python包上传到PyPI，方便其他用户安装和使用。twine是Python包分发过程中不可或缺的工具，提供了简化和安全的上传流程。

### 2.2 发布流程

+ 注册pypi官网账号<br>
[正式注册地址](https://pypi.org/account/register/)、[测试注册地址](https://test.pypi.org/account/register/)<br>

+ 注册成功登录后，生成token用于上传pypi<br>
点击用户名，选择【Account Settings】-> 【API tokens】，完成后，会得到一个很长的以pypi-开头的token。

+ 发布到PyPI-test环境
```
$ ython -m twine upload --repository testpypi dist/*

Uploading distributions to https://test.pypi.org/legacy/
Enter your API token: 
Uploading torchlibrary-0.0.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.2/7.2 kB • 00:00 • ?
Uploading torchlibrary-0.0.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 kB • 00:00 • ?
View at:
https://test.pypi.org/project/torchlibrary/0.0.0/
```

+ 安装
```shell
python -m pip install -i https://test.pypi.org/simple/ torchlibrary==0.0.1
```

+ 测试
```
from torchlibrary import example
print(example.add(1, 2))  # 3
```

## 参考引用

[1] [Python包的构建与发布](https://www.biaodianfu.com/python-package.html)<br>
[2] [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-your-project-to-pypi)<br>