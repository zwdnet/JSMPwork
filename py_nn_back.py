# coding:utf-8
# 《python神经网络编程》实操代码
# 反向查询看看


import numpy as np
import scipy.special
import run
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as pv
import cv2
import glob


# 神经网络类
class NN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置输入、隐藏和输出层维度
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes


        # simple random number
        # self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)

        # Normal distribution
        # average = 0
        # Standard deviation = 1/evolution of number of nodes passed in
        # 用正态分布随机数初始化权重
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 学习率
        self.lr = learningrate

        # 用sigmoid函数做激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        # 激活函数的反函数
        self.inverse_activation_function = lambda x: scipy.special.logit(x)


    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将数据转换为二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 利用传输矩阵wih，计算隐藏层输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出，激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 利用传输矩阵who，计算输出层输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 用激活函数计算输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差值
        output_errors = targets - final_outputs

        # 按权重分配误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        # wj,k = learningrate * error * sigmoid(ok) * (1 - sigmoid(ok)) · oj^T
        # 更新隐藏层及输出层之间的权重值
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        # 更新输入层及隐藏层之间的权重值
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs))


    # 前向传播
    def query(self, inputs_list):
        # 输入矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        # 利用传输矩阵wih，计算隐藏层输入
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        # 计算隐藏层输出，激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        # 利用传输矩阵who，计算输出层输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        
    # 反向查询，给定输出值，看输入会是啥
    def backquery(self, targets_list):
        # 转换为垂直向量
        final_outputs = np.array(targets_list, ndmin = 2).T
        # 计算最后的输入信号，用激活函数的反函数
        final_inputs = self.inverse_activation_function(final_outputs)
        # 计算隐藏层的输出
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # 归一化
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        # 计算进入隐藏层的信号
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # 计算输入层的输出信号
        inputs = np.dot(self.wih.T, hidden_inputs)
        # 归一化
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
        
        
# 加载数据
@run.change_dir
def loadData():
    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    testing_data_file = open("mnist_test.csv", 'r')
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()
    
    return training_data_list, testing_data_list
    
    
# 创建模型
def init_model(input_nodes, hidden_nodes, output_nodes, learning_rate):
    # create instance of neural network
    n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    return n
    
    
# 训练过程
def train(n, epochs, training_data_list, output_nodes):
    # 对训练过程进行循环
    for e in range(epochs):
        print("第{}轮".format(e))
        for record in training_data_list:
            # split the record by the ',' commas
            # 通过','将数分段
            all_values = record.split(',')
            # scale and shift the inputs
            # 将所有的像素点的值转换为0.01-1.00
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
            # creat the target output values
            # 创建标签输出值
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            # 10个输出值，对应的为0.99，其他为0.01
            targets[int(all_values[0])] = 0.99
            # 传入网络进行训练
            n.train(inputs, targets)
    return n
    
    
# 获取预测准确率
def getScores(n, testing_data_list):
    # 创建一个空白的计分卡
    scorecard = []
    # 遍历测试数据
    for record in testing_data_list:
        all_values = record.split(',')
        # 提取正确的标签
        correct_label = int(all_values[0])
        # print(correct_label, 'correct label')
        # 读取像素值并转换
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
        # 通过神经网络得出结果
        outputs = n.query(inputs)
        # 结果
        label = np.argmax(outputs)
        # print(label, "network's answer")
        # 标签相同，计分卡加一，否则加零
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
    # 输出计分卡
    # print(scorecard)
    # 输出分数
    scorecard_array = np.asarray(scorecard)
    
    return scorecard_array
        
        
# 解MINST手写数字识别问题
@run.change_dir
@run.timethis
def minst(trial):
    input_nodes = 784
    hidden_nodes = trial.suggest_categorical("hidden_dim", [50, 100, 200, 300])
    output_nodes = 10
    # 学习率
    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.01, 0.81, 0.1)
    n = init_model(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_list, testing_data_list = loadData()
    # 训练
    epochs = trial.suggest_int("epochs:", 1, 10)
    n = train(n, epochs, training_data_list, output_nodes)
    # 测试
    res = getScores(n, testing_data_list)
    return res.sum() / res.size
    
    
# 画图
@run.change_dir
def draw_results(study):
    # 优化历史
    plt.figure()
    fig = pv.plot_optimization_history(study)
    fig.write_image("./output/opt_his.png")
    plt.close()
    # 等高线图
    plt.figure()
    fig = pv.plot_contour(study)
    fig.write_image("./output/opt_contour.png")
    plt.close()
    # 经验分布图
    plt.figure()
    fig = pv.plot_edf(study)
    fig.write_image("./output/opt_edf.png")
    plt.close()
    # 高维参数
    plt.figure()
    fig = pv.plot_parallel_coordinate(study)
    fig.write_image("./output/opt_coordinate.png")
    plt.close()
    
    
# 手写数字识别应用
# 处理输入数据
@run.change_dir
def data_process():
    targets = []
    datas = []
    for file in glob.glob(r"./pic/*.png"):
        targets.append(int(file.split("/")[2].split(".")[0]))
        img_array = cv2.imread(file)
        img_array = cv2.resize(img_array, (28, 28))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        height,width = img_array.shape
        dst = np.zeros((height,width),np.uint8)
        for i in range(height):
            for j in range(width):
                dst[i,j] = 255 - img_array[i,j]
        img_array = dst.reshape(784)
        datas.append(img_array)
    return (targets, datas)
    
    
# 训练模型
@run.timethis
def trainModel():
    print("开始训练")
    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10
    learning_rate = 0.11
    epochs = 8
    
    model = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_list, _ = loadData()
   
    # 对训练过程进行循环
    for e in range(epochs):
        for record in training_data_list:
            # 通过','将数分段
            all_values = record.split(',')
            # 将所有的像素点的值转换为0.01-1.00
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)
            # 创建标签输出值
            targets = np.zeros(output_nodes) + 0.01
            # 10个输出值，对应的为0.99，其他为0.01
            targets[int(all_values[0])] = 0.99
            # 传入网络进行训练
            model.train(inputs, targets)
            
    return model
    
    
# 用模型识别实际数据
def testModel(model, test_datas, targets):
    n = len(test_datas)
    correct = 0
    for i in range(n):
        # 用模型得出预测值
        outputs = model.query(test_datas[i])
        # 转换为结果
        label = np.argmax(outputs)
        print("预测结果{}，实际结果{}".format(label, targets[i]))
        if label == targets[i]:
            correct += 1
            
    return correct/n
    
    
# 反向查询给定输出的输入
@run.change_dir
def back(model):
    output_nodes = 10
    for i in range(10):
        label = i
        targets = np.zeros(output_nodes) + 0.01
        targets[label] = 0.99
        image_data = model.backquery(targets)
        filename = "./output/"+str(i)+".png"
        print(filename)
        plt.figure()
        plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
        plt.savefig(filename)
        plt.close()
    

if __name__ == "__main__":
    """
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    
    learning_rate = 0.3
    
    # n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # print(n.query([1.0, 0.5, -0.5]))
    
    # minst()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(minst, n_trials=100)
    print("结果:", study.best_params)
    print(study.best_value)
    print(study.best_trial)
    if pv.is_available:
        print("结果作图")
        draw_results(study)
    else:
        print("不能作图")
    """
    # 具体应用模型
    # 目前得到的最佳参数:{'hidden_dim': 300, 'learning_rate': 0.11, 'epochs:': 9}
    # targets, datas = data_process()
    model = trainModel()
    # score = testModel(model, datas, targets)
    # print("模型预测准确率:{}".format(score))
    back(model)
