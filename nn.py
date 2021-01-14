# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 神经网络及深度学习练习


# 先用手撸
# 参考 https://b23.tv/srXty3
from numpy import array, exp, random, dot


# 正向传播
def fp(X, weights):
    z = dot(X, weights)
    return 1/(1+exp(-z))
    
    
# 反向传播
def bp(y, output):
    error = y - output
    return error * output * (1-output)


# 手撸单层神经网络
def nn():
    # X = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    # y = array([[0,1,1,0]]).T
    X = array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    y = array([[0,1,1,0]]).T
    random.seed(1)
    weights = 2*random.random((3,1)) - 1
    for it in range(10000):
        output = fp(X, weights)
        delta = bp(y, output)
        weights += dot(X.T, delta)
    print(weights)
    print(fp([0, 0, 1], weights))
    
    
# 多层正向传播
def mfp(X, w0, w1):
    l1 = 1/(1+exp(-dot(X, w0)))
    l2 = 1/(1+exp(-dot(l1, w1)))
    return l1, l2
    
    
# 反向传播
def mbp(l1, l2, y, w1):
    error = y - l2
    slope = l2 * (1-l2)
    l1_delta = error*slope
    
    l0_error = l1_delta.dot(w1.T)
    l0_slope = l1 * (1-l1)
    l0_delta = l0_error*l0_slope
    return l0_delta, l1_delta
    
    
# 手撸多层神经网络
def mnn():
    # X = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    # y = array([[0,1,1,0]]).T
    X = array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
    y = array([[0,1,1,0]]).T
    random.seed(1)
    # weights = 2*random.random((3,1)) - 1
    w0 = 2*random.random((3, 4)) - 1
    w1 = 2*random.random((4, 1)) - 1
    for it in range(10000):
        l0 = X
        l1, l2 = mfp(X, w0, w1)
        l0_delta, l1_delta = mbp(l1, l2, y, w1)
        w1 += dot(l1.T, l1_delta)
        w0 += dot(l0.T, l0_delta)
    # print(weights)
    print(mfp([0, 0, 0], w0, w1)[1])
    
    
# 再尝试pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


def testTorch():
    # 张量操作
    print("张量操作")
    x = torch.empty(5, 3)
    print(x)
    x = torch.rand(5, 3)
    print(x)
    x = torch.zeros(5, 3, dtype = torch.long)
    print(x)
    x = torch.tensor([5.5, 3])
    print(x)
    x = x.new_ones(5, 3, dtype = torch.double)
    print(x)
    x = torch.randn_like(x, dtype = torch.float)
    print(x)
    print(x.size())
    y = torch.rand(5, 3)
    print(x+y)
    print(torch.add(x, y))
    result = torch.empty(5, 3)
    torch.add(x, y, out = result)
    print(result)
    y.add_(x)
    print(y)
    print(x[:, 1])
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)
    print(x.size(), y.size(), z.size())
    x = torch.randn(1)
    print(x)
    print(x.item())
    # 自动微分
    print("自动微分")
    x = torch.ones(2, 2, requires_grad = True)
    print(x)
    y = x+2
    print(y)
    print(y.grad_fn)
    z = y*y*3
    out = z.mean()
    print(z, out)
    a = torch.randn(2, 2)
    a = ((a*3) / (a-1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)
    b = (a*a).sum()
    print(b.grad_fn)
    out.backward()
    print(x.grad)
    x = torch.randn(3, requires_grad = True)
    y = x*2
    while y.data.norm() < 1000:
        y = y*2
    print(y)
    v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
    y.backward(v)
    print(x.grad)
    print(x.requires_grad)
    print((x**2).requires_grad)
    
    with torch.no_grad():
        print((x**2).requires_grad)
    
    
if __name__ == "__main__":
    nn()
    mnn()
    testTorch()
