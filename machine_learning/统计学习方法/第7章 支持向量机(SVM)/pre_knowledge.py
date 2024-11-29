# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
import math

x = [3,4]

# 欧几里德范数
y = np.linalg.norm(x)
print(y)

# 单位向量，即方向
d = x/y
print(d)

def geometricDotProduct(x,y,theta):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    return x_norm * y_norm * math.cos(math.radians(theta))

print(geometricDotProduct([3,5], [8,2], 45))
print(np.dot([3,5], [8,2]))