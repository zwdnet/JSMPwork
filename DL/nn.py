# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 《深度学习入门:基于python的理论与实现》
# 第三章 神经网络


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
    
    
# 初始化神经网络
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    
    return network
    
    
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
    
    
# 前向传播过程
def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    
    y = identity_function(a3)
    
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
    
    
# 绘制数据
@run.change_dir
def drawNum(data, target):
    i = random.randint(0, data.shape[0]-1)
    print(i)
    img = data[i]
    label = target[i]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    pil_img = Image.fromarray(np.uint8(img*255))
    pil_img.save("./output/number.png", "png")
    
    
# 测试minst
@run.change_dir
def testMinst():
    x_train, t_train, x_test, t_test = loadData()
    drawNum(x_train, t_train)
    
    # 加载训练好的模型
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i+batch_size]
        y_batch = forward(network, x_batch)
        p = np.argmax(y_batch, axis = 1)
        accuracy_cnt += np.sum(p == t_test[i:i+batch_size])
            
    print("预测准确率:{}/{}={}".format(accuracy_cnt, len(x_test), accuracy_cnt/len(x_test)))
    
    

if __name__ == "__main__":
    draw_step()
    draw_sigmoid()
    draw_ReLU()
    
    # 实现神经网络
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
    
    # 测试softmax函数
    a = np.array([1010, 1000, 990])
    y = softmax(a)
    print(y)
    print(np.sum(y))
    
    # minst测试
    testMinst()
        