# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 《深度学习入门:基于python的理论与实现》
# 第五章 误差反向传播法


import numpy as np
import matplotlib.pyplot as plt
import run
import pandas as pd
from PIL import Image
import random
import pickle
from collections import OrderedDict


# 乘法层的实现
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y
        return out
        
    def backward(self, dout):
        dx = dout*self.y # 翻转x和y
        dy = dout*self.x
        
        return dx, dy
        
        
def testMul():
    apple = 100
    apple_num = 2
    tax = 1.1
    
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()
    
    # 前向传播
    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)
    print(price)
    
    # 反向传播
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print(dapple_price, dtax, dapple, dapple_num)
    
    
# 加法层实现
class AddLayer:
    def __init__(self):
        pass
        
    def forward(self, x, y):
        out = x+y
        return out
        
    def backward(self, dout):
        dx = dout*1
        dy = dout*1
        return dx, dy
        
        
def testAdd():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1
    
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()
    
    # 前向传播
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)
    print(price)
    
    # 反向传播
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)
    
    
# ReLU激活函数层
class ReLU:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
        
        
def testReLU():
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    mask = (x<0)
    print(mask)
    relu = ReLU()
    out = relu.forward(x)
    dout = relu.backward(out)
    print(out, dout)
    
    
# Sigmoid激活函数层
class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        
        return out
        
    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        
        return dx
        
        
def testSigmoid():
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    sigmoid = Sigmoid()
    out = sigmoid.forward(x)
    dout = sigmoid.backward(out)
    print(out, dout)
    
    
def testSum():
    print("求和")
    x = np.array([[1, 2], [3, 4]])
    s1 = np.sum(x, axis = 0)
    s2 = np.sum(x, axis = 1)
    s3 = np.sum(x)
    print(x, s1, s2, s3)
    
    
# Affine层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx
        
        
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))
    
    
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        
        
# softmax和loss函数结合层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
        
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx
        
        
# 数值微分
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad


# 用上面这些构建神经网络
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        
        # 生成层
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)
        
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"] )
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"] )
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"] )
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"] )
        
        return grads

        
    #  更快的求梯度的方法
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db
        
        return grads
        
        
# 手写数字识别
# 加载数据
@run.change_dir
def loadData():
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    testing_data_file = open("mnist_test.csv", 'r')
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()
    
    x_train, t_train = [], []
    for record in training_data_list:
        # 通过','将数分段
        all_values = record.split(',')
        # 将所有的像素点的值转换为0.01-1.00
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
        # 创建标签输出值
        target = int(all_values[0])
        x_train.append(inputs)
        t_train.append(target)
    x_test, t_test = [], []
    for record in testing_data_list:
        # 通过','将数分段
        all_values = record.split(',')
        # 将所有的像素点的值转换为0.01-1.00
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
        # 创建标签输出值
        target = int(all_values[0])
        x_test.append(inputs)
        t_test.append(target)
    x_train = np.array(x_train)
    t_train = np.array(t_train)
    x_test = np.array(x_test)
    t_test = np.array(t_test)
    t_train = one_hot(t_train)
    t_test = one_hot(t_test)
    return x_train, t_train, x_test, t_test
    
    
# one_hot过程
def one_hot(t):
    tmp = np.zeros((t.shape[0], 10))
    for i in range(t.shape[0]):
        tmp[i][t[i]] = 1
    t = tmp
    return t
    
        
# 梯度确认
def gradcheck():
    print("梯度确认")
    n = 10
    x_train, t_train, x_test, t_test = loadData()
    network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
    x_batch = x_train[:n]
    t_batch = t_train[:n]
    
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)
    
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + ":" + str(diff))
        
        
# 实际解决手写输入识别问题
@run.change_dir
@run.timethis
def minst():
    print("实际解题")
    x_train, t_train, x_test, t_test = loadData()
    network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
    
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 反向传播求梯度
        grad = network.gradient(x_batch, t_batch)
        
        # 更新参数
        for key in ["W1", "b1", "W2", "b2"]:
            network.params[key] -= learning_rate*grad[key]
            
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("训练集准确率{}，测试集准确率{}".format(train_acc, test_acc))
            
    # 画图
    plt.figure()
    plt.plot(train_loss_list)
    plt.savefig("./output/loss.png")
    plt.close()
    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.savefig("./output/accuracy.png")
    plt.close()
        


if __name__ == "__main__":
    testMul()
    testAdd()
    testReLU()
    testSigmoid()
    testSum()
    gradcheck()
    minst()
        