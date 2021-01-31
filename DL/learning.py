# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 《深度学习入门:基于python的理论与实现》
# 第四章 神经网络的学习


import numpy as np
import matplotlib.pyplot as plt
import run
import pandas as pd
from PIL import Image
import random
import pickle


# 阶跃函数
def step_function(x):
    """
    if x > 0:
        return 1
    else:
        return 0
    """
    # 用支持numpy的形式
    y = x>0
    return y.astype(np.int)
    
    
# 画图
@run.change_dir
def draw_step():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.savefig("./output/step_function.png")
    plt.close()
    
    
# sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
    
# 画图
@run.change_dir
def draw_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.savefig("./output/sigmoid_function.png")
    plt.close()
    
    
# ReLU函数
def ReLU(x):
    return np.maximum(0, x)
    
    
# 画图
@run.change_dir
def draw_ReLU():
    x = np.arange(-5.0, 5.0, 0.1)
    y = ReLU(x)
    plt.plot(x, y)
    plt.savefig("./output/ReLU_function.png")
    plt.close()
    
    
# 恒等函数
def identity_function(x):
    return x
    
    
# softmax函数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #防止数值太大，溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
    
    
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
#    print(x_train.shape)
#    print(t_train.shape)
#    print(x_test.shape)
#    print(t_test.shape)
    return x_train, t_train, x_test, t_test

    
# 均方误差函数
def mse(y, t):
    return 0.5*np.sum((y-t)**2)
    
    
# 交叉熵误差
def cee(y, t):
    delta = 1e-7
    return -np.sum(np.dot(t, np.log(y+delta)))
    
    
# mini-batch选取样本
def mini_batch(x_train, t_train, batch_size):
    train_size = x_train.shape[0]
    assert(train_size >= batch_size)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    return x_batch, t_batch
    
    
# mini_batch版交叉熵误差
def mb_cee(y, t, one_hot = False):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    if one_hot:
        return -np.sum(t*np.log(y+delta))/batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t]+delta))/batch_size
        
        
# 计算f在x处的导数
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(2*h)
    
    
# 定义求导的函数
def function_1(x):
    return 0.01*x**2 + 0.1*x
    
    
def function_2(x):
    return x[0]**2 + x[1]**2
    
    
# 测试数值微分
@run.change_dir
def test_diff():
    # 画图
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.plot(x, y)
    plt.savefig("./output/num_diff.png")
    plt.close()
    
    print(numerical_diff(function_1, 5))
    print(numerical_diff(function_1, 10))
    
    
# (3, 4)时对x0的偏导函数
def function_tmp1(x0):
    return x0**2+4.0**2
    
    
# (3, 4)时对x1的偏导函数
def function_tmp2(x1):
    return 3.0**2+x1**2
    
    
# 测试偏导数
@run.change_dir
def test_pdiff():
    # 画图
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    xx = np.arange(-5.0, 5.0, 0.5)
    yy = np.arange(-5.0, 5.0, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = X**2 + Y**2
    ax1.plot_surface(X, Y, Z)
    plt.savefig("./output/num_pdiff.png")
    plt.close()
    print(numerical_diff(function_tmp1, 3.0))
    print(numerical_diff(function_tmp2, 4.0))
    
   
"""
# 求数值梯度
def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    print(x.shape, x.size)
    for idx in range(x.size):
        print(idx)
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fx1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fx2 = f(x)
        
        grad[idx] = (fx1 - fx2)/(2*h)
        x[idx] = tmp_val
        
    return grad
"""

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad
    
    
def test_grad():
    print(numerical_gradient(function_2, np.array([3.0, 4.0])))
    print(numerical_gradient(function_2, np.array([0.0, 2.0])))
    print(numerical_gradient(function_2, np.array([3.0, 0.0])))
    
    
# 梯度下降法
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    print("学习率{}".format(lr))
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
        
    return x
    
    
# 测试梯度下降法
def test_gd():
    print("测试梯度下降")
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x))
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr = 10.0))
    init_x = np.array([-3.0, 4.0])
    print(gradient_descent(function_2, init_x, lr = 1e-10))
    
    
# 定义简单的神经网络
class simpleNet:
    def __init__(self):
        # 用高斯分布进行初始化
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
        
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cee(y, t)
        return loss
        
        
# 测试神经网络
def test_nn():
    print("测试神经网络")
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    print(net.loss(x, t))
    def f(W):
        return net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(dW)
    
    
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
    
# 两层神经网络
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    def loss(self, x, t):
        y = self.predict(x)
        return cee(y, t)
        
    def accuracy(self, x, t):
        tmp = np.zeros((t.shape[0], 10))+0.01
        for i in range(t.shape[0]):
            tmp[i][t[i]] = 0.99
        t = tmp
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
#        print(type(y), type(t))
#        print(y.shape, t.shape)
#        print(y[0], t[0])
        tmp = np.zeros((t.shape[0], 10))+0.01
        for i in range(t.shape[0]):
            tmp[i][t[i]] = 0.99
        # print(tmp.shape)
        dy = (y - tmp) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
        
        
# 测试两层神经网络
@run.change_dir
@run.timethis
def test_2_nn():
    print("测试两层神经网络")
#    net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
#    print(net.params["W1"].shape)
#    print(net.params["b1"].shape)
#    print(net.params["W2"].shape)
#    print(net.params["b2"].shape)
#    x = np.random.rand(100, 784)
#    y = net.predict(x)
#    # print(y)
#    t = np.random.rand(100, 10)
#    grads = net.numerical_gradient(x, t)
#    print(grads["W1"].shape)
#    print(grads["b1"].shape)
#    print(grads["W2"].shape)
#    print(grads["b2"].shape)
    
    # 加载数据
    x_train, t_train, x_test, t_test = loadData()
    
    # 训练
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    # 超参数
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    # 平均每个epoch的重复次数
    iter_per_epoch = max(train_size/batch_size, 1)

    network = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
    
    for i in range(iters_num):
        # 获取mini_batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 计算梯度
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)
        # 更新参数
        for key in ["W1", "b1", "W2", "b2"]:
            network.params[key] -= learning_rate*grad[key]
        
        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # print(i, loss)
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
    # 测试mse
    t = np.zeros(10)
    t[2] = 1
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print(t, y)
    print(mse(y, t))
    print(cee(y, t))
    
    # 测试mini_batch
    x_train, t_train, x_test, t_test = loadData()
    x_batch, t_batch = mini_batch(x_train, t_train, 10)
    print(x_batch)
    print(t_batch)
    
    # 数值微分
    test_diff()
    test_pdiff()
    test_grad()
    test_gd()
    
    # 测试神经网络
    test_nn()
    # 测试两层神经网络
    test_2_nn()
    
        