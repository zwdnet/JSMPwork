# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 《深度学习入门:基于python的理论与实现》
# 第六章 与学习相关的技巧


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
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, init_method = "std"):
        self.init_method = init_method
        # 初始化权重
        self.params = {}
        if init_method == "std":
            self.params["W1"] = weight_init_std*np.random.randn(input_size, hidden_size)*0.01
            self.params["W2"] = weight_init_std*np.random.randn(hidden_size, output_size)*0.01
        elif init_method == "Xavier":
            self.params["W1"] = weight_init_std*np.random.randn(input_size, hidden_size)/np.sqrt(input_size)
            self.params["W2"] = weight_init_std*np.random.randn(hidden_size, output_size)/np.sqrt(hidden_size)
        else:
            self.params["W1"] = weight_init_std*np.random.randn(input_size, hidden_size)/np.sqrt(2/input_size)
            self.params["W2"] = weight_init_std*np.random.randn(hidden_size, output_size)/np.sqrt(2/hidden_size)                
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)
        
        # 生成层
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        if self.init_method == "std":
            self.layers["Sigmoid1"] = Sigmoid()
        else:
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
def minst(init_method = "std"):
    print("实际解题")
    x_train, t_train, x_test, t_test = loadData()
    network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10, init_method = init_method)
    
    iters_num = 1000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    optimizer = Adam()
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # 反向传播求梯度
        grad = network.gradient(x_batch, t_batch)
        
        # 更新参数
        #for key in ["W1", "b1", "W2", "b2"]:
#            network.params[key] -= learning_rate*grad[key]
        optimizer.update(network.params, grad)
            
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
    
    return train_loss_list
        
        
# 实现SGD
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]
            
            
# 实现Momentum
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = 0.9
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]
            
            
# 实现AdaGrad
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 实现Adam
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            
# 实现sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
            
            
# 隐藏层激活值的分布实验
@run.change_dir
def hidden_value():
    x = np.random.randn(1000, 100)
    node_num = 100
    # xavier初始值
    w = np.random.randn(node_num, node_num)/np.sqrt(2.0/node_num)
    hidden_layer_size = 5
    activations = {}
    
    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]
            
        # w = np.random.randn(node_num, node_num)*0.01
        
        z = np.dot(x, w)
        # a = sigmoid(z)
        # a = np.tanh(x)
        a = np.maximum(0, x)
        activations[i] = a
        
    # 画图
    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1)+"-layers")
        plt.hist(a.flatten(), 30, range = (0, 1))
    plt.savefig("./output/initparams6.png")
    
    
# 测试不同的权重初始化方法
@run.change_dir
def test_init():
    init_methods = ["std", "Xavier", "He"]
    results = []
    for method in init_methods:
        results.append(minst(method))
        
    plt.figure()
    for i in range(3):
        plt.plot(results[i], label = init_methods[i])
    plt.legend()
    plt.savefig("./output/init_methods2.png")
    plt.close()
    
    
# 多层神经网络
class MultiLayerNet:
    """全连接的多层神经网络

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
            self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """求损失函数

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        损失函数的值
        """
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
    
    
# 测试过拟合
@run.change_dir
@run.timethis
def testoverfit():
    print("过拟合")
    x_train, t_train, x_test, t_test = loadData()
    # 人为减少训练集大小
    x_train = x_train[:300]
    t_train = t_train[:300]
    network = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], output_size = 10)
    optimizer = SGD(lr = 0.01)
    
    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        train_loss_list.append(network.loss(x_batch, t_batch))
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
        
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("迭代次数:{}，训练集准确率{}，测试集准确率{}".format(i, train_acc, test_acc))
            
    # 画图
    plt.figure()
    plt.plot(train_loss_list)
    plt.savefig("./output/overfit_loss.png")
    plt.close()
    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.savefig("./output/overfit_accuracy.png")
    plt.close()
    
    
# 测试用权值衰减解决过拟合
@run.change_dir
@run.timethis
def decay():
    print("权值衰减")
    x_train, t_train, x_test, t_test = loadData()
    # 人为减少训练集大小
    x_train = x_train[:300]
    t_train = t_train[:300]
    # 衰减比率
    weight_decay_lambda = 0.1
    network = MultiLayerNet(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], output_size = 10, weight_decay_lambda = weight_decay_lambda)
    optimizer = SGD(lr = 0.01)
    
    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        train_loss_list.append(network.loss(x_batch, t_batch))
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
        
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("迭代次数:{}，训练集准确率{}，测试集准确率{}".format(i, train_acc, test_acc))
            
    # 画图
    plt.figure()
    plt.plot(train_loss_list)
    plt.savefig("./output/decay_overfit_loss.png")
    plt.close()
    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.savefig("./output/decay_overfit_accuracy.png")
    plt.close()
    
    
# Dropout类
class Dropout:
    def __init__(self, dropout_rate = 0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        
    def forward(self, x, train_flag = True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x*self.mask
        else:
            return x*(1.0 - self.dropout_rate)
            
    def backward(self, dout):
        return dout*self.mask
        
        
class MultiLayerNetExtend:
    """扩展版的全连接的多层神经网络
    
    具有Weiht Decay、Dropout、Batch Normalization的功能

    Parameters
    ----------
    input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
    use_dropout: 是否使用Dropout
    dropout_ration : Dropout的比例
    use_batchNorm: 是否使用Batch Normalization
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设定权重的初始值

        Parameters
        ----------
        weight_init_std : 指定权重的标准差（e.g. 0.01）
            指定'relu'或'he'的情况下设定“He的初始值”
            指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """求损失函数
        参数x是输入数据，t是教师标签
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        """求梯度（数值微分）

        Parameters
        ----------
        X : 输入数据
        T : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads
        
        
# 测试dropout
@run.change_dir
@run.timethis
def test_dropout():
    print("dropout")
    x_train, t_train, x_test, t_test = loadData()
    # 人为减少训练集大小
    x_train = x_train[:3000]
    t_train = t_train[:3000]
    
    use_dropout = True
    dropout_ratio = 0.2
    
    network = MultiLayerNetExtend(input_size = 784, hidden_size_list = [100, 100, 100, 100, 100, 100], output_size = 10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
    optimizer = SGD(lr = 0.01)
    
    max_epochs = 201
    train_size = x_train.shape[0]
    batch_size = 100
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size/batch_size, 1)
    epoch_cnt = 0
    
    for i in range(3000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        train_loss_list.append(network.loss(x_batch, t_batch))
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
        
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("迭代次数:{}，训练集准确率{}，测试集准确率{}".format(i, train_acc, test_acc))
            
    # 画图
    plt.figure()
    plt.plot(train_loss_list)
    plt.savefig("./output/dropout_loss.png")
    plt.close()
    plt.figure()
    plt.plot(train_acc_list)
    plt.plot(test_acc_list)
    plt.savefig("./output/dropout_accuracy.png")
    plt.close()
    

if __name__ == "__main__":
    #testMul()
#    testAdd()
#    testReLU()
#    testSigmoid()
#    testSum()
#    gradcheck()
    # minst()
    # hidden_value()
    # test_init()
    # testoverfit()
    # decay()
    test_dropout()
    pass
        