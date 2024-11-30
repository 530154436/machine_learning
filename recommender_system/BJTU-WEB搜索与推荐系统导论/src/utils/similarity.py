#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: zhengchubin
# @time: 2021/5/6 20:38
# @function： 集合间相似度
#            https://www.yuque.com/malianhong/vgi1sh/vfoscm
#            https://mieruca-ai.com/ai/jaccard_dice_simpson/
from typing import Union
from collections import Counter
import numpy as np


def jaccard(a, b) -> float:
    """
    杰卡德相似系数 Jaccard = |A&B|/|AUB|

    Dice系数与Jaccard非常的类似。Jaccard是在分子和分母上都减去了A∩B。
    Args:
        a:  集合A
        b:  集合B
    """
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return 0.0
    return len(a & b)/len(a | b)


def dice(a, b):
    """
    骰子系数 dice = 2*|A&B|/(|A|+|B|)
    Args:
        a:  集合A
        b:  集合B
    """
    a, b = set(a), set(b)
    return 2*len(a & b)/(len(a)+len(b))


def simpson(a, b):
    """
    辛普森系数  simpson = |A&B|/min(|A|, |B|)
    Args:
        a:  集合A
        b:  集合B
    """
    a, b = set(a), set(b)
    return len(a & b)/min(len(a), len(b))


def cosine_bow(x: Union[list, tuple], y: Union[list, tuple]):
    """
    词袋模型 计算余弦相似度
    Args:
        x:      array like
        y:      array like
    """
    # 全集
    all_data = set(x) | set(y)

    # 分别对x、y计数
    x_counter, y_counter = Counter(x), Counter(y)

    # 向量化
    x_vec, y_vec = np.zeros(len(all_data)), np.zeros(len(all_data))
    for i, item in enumerate(all_data):
        x_vec[i] = x_counter.get(item, 0)
        y_vec[i] = y_counter.get(item, 0)

    # 计算余弦相似度
    inner = np.dot(x_vec, y_vec)
    x_l2, y_l2 = np.linalg.norm(x_vec, ord=2), np.linalg.norm(y_vec, ord=2)

    return np.divide(inner, x_l2*y_l2)


def hamming_distance(x: int, y: int, f: int = 64):
    """
    汉明距离
    二进制数减去1，则二进制串中的最后一个1及后面的数字全都反了，
    而n & (n-1)就相当于把后面的数字清0
    :param x:   整数x
    :param y:   整数y
    :param f:   二进制串位数
    """
    x = (x ^ y) & ((1 << f) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ans


def hamming_sim(x: int, y: int, f: int = 64):
    """
    汉明距离的相似度
    二进制数减去1，则二进制串中的最后一个1及后面的数字全都反了，
    而n & (n-1)就相当于把后面的数字清0
    :param x:   整数x
    :param y:   整数y
    :param f:   二进制串位数
    """
    return 1 - hamming_distance(x, y, f) / f


def hamming_sim_v2(x: np.array, y: np.array) -> float:
    """
    汉明距离
    """
    assert x.shape == y.shape
    distance, n = np.sum(x ^ y), x.shape[0]
    return 1 - np.sum(distance) / n


if __name__ == '__main__':
    s = cosine_bow(['1', '1', '3', '1', '2'], ['4', '1', '3', '4', '2'])
    print(s)

    print(hamming_distance(3, 8))
