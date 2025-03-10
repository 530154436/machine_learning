torch和torchvision对应版本
https://github.com/pytorch/vision#installation

+ 赛题地址
```
一点资讯：https://tech.yidianzixun.com/competition/#/user/problems
BJTU：http://202.112.156.14:3000/plat/
https://www.logicjake.xyz/2021/09/20/一点资讯技术编程大赛CTR赛道-赛后总结/
```


+ M2芯片装lightgbm（巨坑）
```shell
/usr/sbin/softwareupdate --install-rosetta --agree-to-license
arch -x86_64 /bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
arch -x86_64 brew install libomp
ln -s /usr/local/Cellar/libomp/15.0.2/lib/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib
```

+ dask
https://dask.pydata.org/en/latest/dataframe.html
A Dask DataFrame is a large parallel DataFrame composed of many smaller pandas DataFrames, split along the index. These pandas DataFrames may live on disk for larger-than-memory computing on a single machine, or on many different machines in a cluster. One Dask DataFrame operation triggers many operations on the constituent pandas DataFrames.