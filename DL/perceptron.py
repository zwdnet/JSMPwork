# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 《深度学习入门:基于python的理论与实现》
# 第二章 感知机


import numpy as np


# 用感知机来实现逻辑门
# 与门
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1*x1 + w2*x2
    if tmp <= theta:
        return 0
    else:
        return 1
        
        
# 另一种形式实现与门 b = -theta
def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1
        
       
# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1
        
        
# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1
        
        
# 用感知机组合实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND2(s1, s2)
    return y
    

if __name__ == "__main__":
    print(AND(0, 0), AND(0, 1), AND(1, 0), AND(1, 1))
    print(AND2(0, 0), AND2(0, 1), AND2(1, 0), AND2(1, 1))
    print(NAND(0, 0), NAND(0, 1), NAND(1, 0), NAND(1, 1))
    print(OR(0, 0), OR(0, 1), OR(1, 0), OR(1, 1))
    print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))
    