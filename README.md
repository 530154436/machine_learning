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

#### 1. 树模型(by Numpy)
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

#### 2. 矩阵分解(MF by Numpy)
- [x] SVD (统计学习方法第15章 pureSVD)
- [x] 实现Netflix Prize FunkSVD ( Latent Factor Model(LFM))
- [x] BiasSVD (加入偏置项后的 Funk-SVD)
- [x] 实现Koren's SVD++
- [x] NMF(非负矩阵分解)

#### 3. FM-系列: 论文阅读与复现(TF)
- [x] FM ：《Factorization Machines》、《Factorization Machines with libFM》
- [ ] FFM ：《Field-aware Factorization Machines for CTR Prediction》
- [ ] FNN：Factorisation Machine supported Neural Network)
- [ ] DeepFM ：《DeepFM: A Factorization-Machine based Neural Network for CTR Prediction》
- [ ] XDeepFM ：
- [ ] NFM ：《Neural Factorization Machines for Sparse Predictive Analytics》
- [ ] AFM ：《Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks》

#### 4. Embedding-系列: 论文阅读与复现(TF)
- [ ]
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 


- [ ] GBDT + LR ：《Practical Lessons from Predicting Clicks on Ads at Facebook》
- [ ] Wide & Deep ：《Wide & Deep Learning for Recommender Systems》
- [ ] DCN ：《Deep & Cross Network for Ad Click Predictions》
- [ ] MLR ：《Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction》
- [ ] DIN ：《Deep Interest Network for Click-Through Rate Prediction》
- [ ] DIEN ：《Deep Interest Evolution Network for Click-Through Rate Prediction》
- [ ] BPR ：《BPR: Bayesian Personalized Ranking from Implicit Feedback》
- [ ] Youtube ：《Deep Neural Networks for YouTube Recommendations》

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
