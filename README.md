## statistical-learning-method
## 李航《统计学习方法》-基于Python算法实现
### 简介
《统计学习方法》，作者李航，本书全面系统地介绍了统计学习的主要方法，特别是监督学习方法，包括感知机、k近邻法、朴素贝叶斯法、决策树、逻辑斯谛回归与最大熵模型、支持向量机、提升方法、EM算法、隐马尔可夫模型和条件随机场等。除第1章概论和最后一章总结外，每章介绍一种方法。

### 目录
- [x] 第1章 统计学习方法概论
- [x] 第2章 感知器模型(Perceptron)
- [ ] 第3章 K近邻法(KNearestNeighbors)
- [x] 第4章 朴素贝叶斯(NaiveBayes)
- [x] 第5章 决策树(DecisonTree)
- [x] 第6章 逻辑斯提回归与最大熵模型(LogisticRegression)
- [ ] 第7章 支持向量机(SVM)
- [x] 第8章 提升方法(AdaBoost)
- [ ] 第9章 EM算法(EM)
- [ ] 第10章 隐马尔科夫模型(HMM)
- [ ] 第11章 条件随机场(CRF)
- [ ] 第12章 统计学习方法总结
- [ ] 第13章 无监督学习概论
- [ ] 第14章 聚类方法
- [x] 第15章 奇异值分解(SVD)
- [ ] 第16章 主成分分析(PCA)
- [ ] 第17章 潜在语义分析
- [ ] 第18章 概率潜在语义分析
- [ ] 第19章 马尔可夫链蒙特卡罗法
- [ ] 第20章 潜在狄利克雷分配
- [ ] 第21章 PageRank算法
- [ ] 第22章 无监督学习方法总结

#### 1. 树模型
- [x] 实现决策树ID3、C4.5算法
- [x] 实现Cart(回归树)
- [x] 预/后剪枝算法
- [ ] 实现Cart模型树(顺便实现线性回归)
- [x] 实现AdaBoost算法
- [x] 回归问题的提升树算法
- [x] 梯度提升(GBDT)算法
- [x] 实现随机森林(西瓜书8.3 p178)
- [x] 了解、尝试实现XGBoost
- [ ] 整合XGBoost、LightGBM、CatBoost三个工具包


#### 2. 矩阵分解(MF)
- [x] SVD (统计学习方法第15章 pureSVD)
- [x] 实现Netflix Prize FunkSVD ( Latent Factor Model(LFM))
- [x] BiasSVD (加入偏移项后的 Funk-SVD)
- [ ] PMF(概率矩阵分解)，FunkSVD的概率解释版本
- [ ] 实现Koren's SVD++
- [ ] NMF(非负矩阵分解)

> https://zhuanlan.zhihu.com/p/35262187
首先因为低秩假设，一个用户可能有另外一个用户与他线性相关（物品也一样），所以用户矩阵完全可以用一个比起原始UI矩阵更低维的矩阵表示，pureSVD就可降维得到两个低维矩阵，但是此方法要求原始矩阵稠密，因此要填充矩阵（只能假设值），因此有了funkSVD直接分解得到两个低维矩阵。
因为用户,物品的偏置爱好问题所以提出了biasSVD。
因为用户行为不仅有评分，且有些隐反馈（点击等），所以提出了SVD++。
因为假设用户爱好随时间变化，所以提出了timeSVD。
因为funkSVD分解的两个矩阵有负数，现实世界中不好解释，所以提出了NMF。
为了符合TopN推荐，所以提出了WMF。推翻低秩假设，提出了LLORMA（局部低秩）。
因为以上问题都未解决数据稀疏和冷启动问题，所以需要用上除了评分矩阵之外的数据来使推荐更加丰满，即加边信息。

#### 3. FM模型
- [ ] 实现FM
- [ ] 实现FFM
- [ ] 实现DeepFM
- [ ] 实现NFM
- [ ] 实现AFM
- [ ] 实现xDeepFM


推荐系统中的矩阵分解技术 http://www.52nlp.cn/juzhenfenjiedatagrand
树模型：GBDT，XGBoost，LightGBM，CatBoost，NGBoost
Attention模型：DIN，DIEN，DSIN，Transformer，BERT
Embedding：Word2vec，DeepWalk, Node2Vec，GCN
时间序列：AR, MA, ARMA ,ARIMA, LSTM

推荐系统 http://xtf615.com/2018/05/03/recommender-system-survey/

### 引用
>[1]https://github.com/wzyonggege/statistical-learning-method

>[2]https://github.com/WenDesi/lihang_book_algorithm

>[3]https://github.com/fengdu78/lihang-code
