# /usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np
'''
    软间隔最大化 (线性不可分支持向量机)
'''
w = np.array([0.4, 1])
b = -10
x = np.array([6, 8])
y = -1
def constraint(w, b, x, y):
    return y * (np.dot(w, x) + b)
def hard_constraint_is_satisfied(w, b, x, y):
    return constraint(w, b, x, y) >= 1
def soft_constraint_is_satisfied(w, b, x, y, zeta):
    return constraint(w, b, x, y) >= 1 - zeta
# While the constraint is not satisfied for the example (6,8).
print(hard_constraint_is_satisfied(w, b, x, y)) # False
# We can use zeta = 2 and satisfy the soft constraint.
print(soft_constraint_is_satisfied(w, b, x, y, zeta=2)) # True