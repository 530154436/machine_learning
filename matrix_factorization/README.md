[TOC]
#### 1. 传统的SVD算法
$$
M_{m \times n)}=U_{m \times k}\Sigma_{k \times k }V^T_{k \times n}
$$
#### 2. Funk-SVD
2006年，Simon Funk在博客上公开发表了一个只考虑已有`评分`记录的矩阵分解方法，称为`Funk-SVD`，即被Yehuda Koren称为`隐语义模型`(LFM)的矩阵分解方法。
它的出发点为，既然将一个矩阵做SVD分解成3个矩阵很耗时，同时还面临稀疏的问题，那么我们能不能避开稀疏问题，同时只分解成两个矩阵呢？即期望我们的矩阵M这样进行分解：
$$
M_{(m,n)}=P_{m\times k}  Q_{n \times k}^T
$$
+ `矩阵P` $(u,k)$：用户 $u$ 对特征 $k$ 的偏好程度.

+ `矩阵Q` $(j,k)$：物品 $j$ 拥有特征 $k$ 的程度.



采用`线性回归`的思想，目标是使用户真实评分 ($r_{u,j}$) 与矩阵乘积(预测函数)的`残差`尽可能的小。其中，预测函数表示用户 $u$ 对物品 $j$ 的偏好，即
 $$
\hat{r}_{u,j}=p_u q_j^T
$$
用`均方差`表示损失函数
$$
\arg  \underset {p_uq_j} {\min} = \sum \frac 1 2 (r_{u,j} - \hat{r}_{u,j})^2
$$
为防止过拟合，加入一个 $L_2$ 的`正则化项`，因此 Funk-SVD的优化目标函数为: 
$$
\arg  \underset {p_uq_j} {\min} \sum_{u=1}^m\sum_{j=1}^n \frac 1 2 (r_{u,j}-\hat{r}_{u,j})^2 +\frac \lambda  2(||p_u||^2_2+||q_j||^2_2)
$$
+ $m_{u,j}$ 为用户 $u$ 对物品 $j$ 的真实评分

+ $\lambda$ 是正则化系数，需要调参。



评分矩阵为 $M_{m \times n}$，通过直接优化以上损失函数得到用户特征矩阵 $P(m \times k)$ 和物品特征矩阵$Q( n \times k)$，其中 $k<<m,n$。对于这个优化问题，一般通过`梯度下降法`来进行优化得到结果。

将上式分别对 $p_u$ , $q_j$求导，得：

$$
\frac{ \partial J}{ \partial  p_u}=-(r_{uj}-\hat{r}_{uj})q_j+\lambda p_u \\
\frac{ \partial J}{\partial  q_j}=-(r_{uj}-\hat{r}_{uj})p_u+\lambda q_j
$$
在梯度下降法迭代时，$p_u$,$q_j$ 的迭代公式为 ($\eta:学习率$)：
$$
p_u=p_u + \eta [(r_{uj}-\hat{r}_{uj})q_j-\lambda p_u] \\
q_j=q_j+\eta[(r_{uj}-\hat{r}_{uj})p_u-\lambda q_j ]
$$
+ 梯度下降
```python
def sgd(self, u, j, y_true):
    '''
    梯度下降更新参数
    '''
    err = y_true - self.predict(u, j)
    self.P[u] += self.learning_rate * (err * self.Q[j] - self._lambda * self.P[u])
    self.Q[j] += self.learning_rate * (err * self.P[u] - self._lambda * self.Q[j])
```
+ 模型预测
```python
def _know_u(self, u:int):
    return u!=None and u>=0 and u<self.P.shape[0]

def _know_j(self, j:int):
    return j!=None and j>=0 and j<self.Q.shape[0]

def predict(self, u:int, j:int):
    '''
    预测u用户对物品i的评分
    :param u:   用户
    :param j:   物品
    '''
    if self._know_u(u) and self._know_j(j):
        return np.dot(self.P[u, :], self.Q[j, :])
    else:
        return None
```
#### 2. Bias-SVD
BiasSVD矩阵分解主要是在正则化项中加入偏置约束。这些约束都是由独立于用户或物品的因素组成，与用户对物品的偏好无关。
+ 个性化部分：用户和物品的交互，即用户对物品的喜好程度
+ `偏置(Bias)`部分： 独立于用户或独立于物品的因素。主要由三个子部分组成，分别是
    - 训练集中所有评分记录的全局平均数 $\mu$，表示了训练数据的总体评分情况
    - 用户偏置 $b_u$，表示某一特定用户的打分习惯。例如乐观型用户则打分比较保守，总体打分要偏高。
    - 物品偏置 $b_j$，表示某一特定物品得到的打分情况。例如好电影获得的总体评分偏高。

综上，偏置部分可以表示为
$$
b_{uj} = \mu + b_u + b_j
$$
预测评分函数表示为
$$
\hat{r}_{u,j}=p_u q_j^T + b_{uj} = p_u q_j^T + \mu + b_u + b_j
$$
从而优化目标函数 $J(p,q, b_u, b_j)$ ($r_{uj}=m_{uj}$)表示为:
$$
\arg \min \sum_{r_{uj} \in R_{train}} \frac 1 2 \left(r_{uj} - \hat{r}_{uj} \right)^2 + \frac {\lambda} 2 \left(b_u^2 + b_j^2 + ||q_j||^2 + ||p_u||^2\right)
$$
将上式分别对 $p_u、q_j、b_u、b_j$求导，得：

$$
\begin{align*}
\frac{ \partial J}{ \partial  p_u} &=-(r_{uj}-\hat{r}_{uj})q_j+\lambda p_u \\
\frac{ \partial J}{\partial  q_j} &=-(r_{uj}-\hat{r}_{uj})p_u+\lambda q_j \\
\frac{ \partial J}{\partial b_u} &=-(r_{uj}-\hat{r}_{u,j})+\lambda b_u \\ 
\frac{ \partial J}{\partial b_j} &=-(r_{uj}-\hat{r}_{u,j})+\lambda b_j
\end{align*}
$$
在梯度下降法迭代时，$p_u、q_j、b_u、b_j$ 的迭代公式为 ($\eta:学习率$)：
$$
\begin{align*}
p_u &=p_u + \eta [(r_{uj}-\hat{r}_{uj})q_j-\lambda p_u] \\
q_j &=q_j+\eta[(r_{uj}-\hat{r}_{uj})p_u-\lambda q_j ] \\
b_u &=b_u+\eta[(r_{uj}-\hat{r}_{u,j})-\lambda b_u ] \\
b_j &=b_j+\eta[(r_{uj}-\hat{r}_{u,j})-\lambda b_j] \\
\end{align*}
$$
+ 梯度下降
```python
def sgd(self, u, j, y_true):
    '''
    梯度下降更新参数
    '''
    e_uj = y_true - self.predict(u, j)
    self.P[u] += self.learning_rate * (e_uj * self.Q[j] - self._lambda * self.P[u])
    self.Q[j] += self.learning_rate * (e_uj * self.P[u] - self._lambda * self.Q[j])
    self.bu[u] += self.learning_rate * (e_uj - self._lambda * self.bu[u])
    self.bj[j] += self.learning_rate * (e_uj - self._lambda * self.bj[j])
```
+ 预测部分
```python
def predict(self, u:int, j:int):
    '''
    预测u用户对物品i的评分
    '''
    rating = self.mu
    know_u = self._know_u(u)
    know_j = self._know_j(j)
    if know_u: rating += self.bu[u]
    if know_j: rating += self.bj[j]
    if know_u and know_j:
        rating += np.dot(self.P[u, :], self.Q[j, :])
    return rating
```
#### 4. SVD++
后来又提出了对BiasSVD改进的SVD++。它是基于这样的假设：除了显示的评分行为以外，用户对于商品的`浏览记录`或`购买记录`（隐式反馈）也可以从侧面反映用户的偏好。相当于引入了额外的信息源，能够解决因显示评分行为较少导致的冷启动问题。其中一种主要信息源包括：用户 $u$ 产生过行为(显示或隐式)的商品集合 $N(u)$ , 可以引入用户 $u$ 对于这些商品的隐式偏好 $y_i$。
$y_i$ 是隐藏的、对于商品 $i$ 的个人喜好偏置（相当于每种产生行为的商品都有一个偏好 $y_i$）。并且 $y_i$ 是一个向量 (维度=商品数 $\cdot$ 隐因子个数)，每个分量代表对该商品的某一隐因子成分的偏好程度。 而用户 $u$ 对这些隐因子的偏好程度 (implicit feedback) 实际上是将`所有产生行为`的商品对应的隐因子分量值进行分别求和，并除以一个规范化因子 $ \sqrt{ |N_u|}$，其中，引入 $ \sqrt{ |N_u|}$  是为了消除不同 $|N(u)|$ 个数引起的差异。  即
$$
\text{ifb}_u = \frac { \underset {i \in N(u)} \sum  y_i} {\sqrt{ |N_u|} }
$$
预测评分函数表示为
$$
\begin{align*}
b_{uj} &= \mu + b_u + b_j \\
\hat{r}_{u,j}&= (p_u + \text{ifb}_u )q_j^T + b_{uj} = (p_u + \frac { \underset {i \in N(u)} \sum  y_i} {\sqrt{ |N_u|} })q_j^T + \mu + b_u + b_j
\end{align*}
$$
从而优化目标函数 $J(p,q, b_u, b_j, y_i)$ 表示为:
$$
\arg \min \sum_{r_{uj} \in R_{train}} \frac 1 2 \left(r_{uj} - \hat{r}_{uj} \right)^2 + \frac {\lambda} 2 (b_u^2 + b_j^2 + ||q_j||^2 + ||p_u||^2 + \underset {i \in N(u)} \sum  ||y_i||^2)
$$
将上式分别对 $p_u、q_j、b_u、b_j、\underset {i \in N(u)} y_i$求导，得：

$$
\begin{align*}
\frac{ \partial J}{ \partial  p_u} &=-(r_{uj}-\hat{r}_{uj})q_j+\lambda p_u \\
\frac{ \partial J}{\partial  q_j} &=-(r_{uj}-\hat{r}_{uj})p_u+\lambda q_j \\
\frac{ \partial J}{\partial b_u} &=-(r_{uj}-\hat{r}_{u,j})+\lambda b_u \\ 
\frac{ \partial J}{\partial b_j} &=-(r_{uj}-\hat{r}_{u,j})+\lambda b_j \\
\frac{ \partial J} {\partial  y_i} &=- \frac {(r_{uj}-\hat{r}_{u,j}) q_j} {\sqrt{ |N_u|} }  +\lambda y_i
\end{align*}
$$
在梯度下降法迭代时，$p_u、q_j、b_u、b_j、\underset {i \in N(u)} y_i$ 的迭代公式为 ($\eta:学习率$)：
$$
\begin{align*}
p_u &=p_u + \eta [(r_{uj}-\hat{r}_{u,j})q_j-\lambda p_u] \\
q_j &=q_j+\eta[(r_{uj}-\hat{r}_{u,j})p_u-\lambda q_j ] \\
b_u &=b_u+\eta[(r_{uj}-\hat{r}_{u,j})-\lambda b_u ] \\
b_j &=b_j+\eta[(r_{uj}-\hat{r}_{u,j})-\lambda b_j] \\
y_i &= y_i + \frac {(r_{uj}-\hat{r}_{u,j}) q_j} {\sqrt{|N_u|}}  - \lambda y_i
\end{align*}
$$
+ 梯度下降
```python
def sgd(self, u, j, y_true):
    '''
    梯度下降更新参数
    '''
    # 残差
    e_uj = y_true - self.predict(u, j)

    # 更新显示因子
    self.P[u] += self.learning_rate * (e_uj * self.Q[j] - self._lambda * self.P[u])
    self.Q[j] += self.learning_rate * (e_uj * self.P[u] - self._lambda * self.Q[j])

    # 更新偏置
    self.bu[u] += self.learning_rate * (e_uj - self._lambda * self.bu[u])
    self.bj[j] += self.learning_rate * (e_uj - self._lambda * self.bj[j])

    # 更新隐式因子
    ui = self.ui[u]
    ui_sqrt = np.sqrt(len(ui))
    self.yi[ui] = self.learning_rate * (e_uj * self.Q[j] / ui_sqrt - self._lambda * self.yi[ui])
    # for i in ui:
    #     self.yi[i] = self.learning_rate * (e_uj * self.Q[j] / ui_sqrt - self._lambda * self.yi[i])
    self.u_implicit_fb[u] = np.sum(self.yi[ui], axis=0) / ui_sqrt
```
+ 预测函数
```python
def predict(self, u:int, j:int):
    '''
    预测u用户对物品i的评分
    '''
    rating = self.mu
    know_u = self._know_u(u)
    know_j = self._know_j(j)
    if know_u:
        rating += self.bu[u]
    if know_j:
        rating += self.bj[j]
    if know_u and know_j:
        rating += np.dot(self.P[u, :] + self.u_implicit_fb[u], self.Q[j, :])
    return rating
```
#### 5. NMF
#### 6. 总结与参考链接
|算法 | 别名 | 内容|
|:---|:---|:---|
|SVD | traditional SVD | 奇异值分解|
|FunkSVD | LFM, basic MF, MF | LFM|
|regularized SVD | regularized MF | LFM+正则项|
|bias SVD | bias MF | LFM+正则项+偏置项|
|SVD++ | * | LFM+正则项+偏置项+隐性反馈|
|NMF | * | 对隐向量非负限制，可用在bias SVD等不同模型上|

- [1] [奇异值分解(SVD)原理与在降维中的应用](https://www.cnblogs.com/pinard/p/6251584.html)
- [2] [Simon-Funk的博客](https://sifter.org/~simon/journal/20061211.html)
- [3] [推荐系统-矩阵分解技术](http://freewill.top/2017/03/07/机器学习算法系列（13）：推荐系统（3）——矩阵分解技术/)
- [4] [Surprise框架MF的实现](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)
- [5] [基于矩阵分解的协同过滤](https://mathpretty.com/11495.html)
- [6] [推荐系统算法调研(理论讲解Nice)](http://xtf615.com/2018/05/03/recommender-system-survey/)