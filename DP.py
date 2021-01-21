# coding:utf-8
# kaggle Jane Street Market Prediction代码
"""深度学习练习代码，参考
https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from run import *

from sklearn.model_selection import train_test_split


# 深度学习基础

# 加载数据
@change_dir
def loadData():
    x_1 = np.load("./X.npy")
    y_1 = np.load("./Y.npy")
    img_size = 64
    plt.subplot(1, 2, 1)
    plt.imshow(x_1[260].reshape(img_size, img_size))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(x_1[900].reshape(img_size, img_size))
    plt.axis("off")
    plt.savefig("./output/data.png")
    # 把数据连接起来，并创建标签
    X = np.concatenate((x_1[204:409], x_1[822:1027]), axis = 0)
    z = np.zeros(205)
    o = np.ones(205)
    Y = np.concatenate((z, o), axis = 0).reshape(X.shape[0], 1)
    print(X.shape)
    print(Y.shape)
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 42)
    number_of_train = X_train.shape[0]
    number_of_test = X_test.shape[0]
    #将三维数据变换到二维
    X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[2])
    X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[2])
    print("X_train_flatten", X_train_flatten.shape)
    print("X_test_flatten", X_test_flatten.shape)
    # 将数据倒置
    x_train = X_train_flatten.T
    x_test = X_test_flatten.T
    y_train = Y_train.T
    y_test = Y_test.T
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # 返回数据
    return (x_train, y_train, x_test, y_test)
    
    
# 初始化参数
def init_params(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b
    
 
# 定义sigmoid函数
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
    
    
# 前向传播过程
def fp(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head)
    # 平均成本
    cost = (np.sum(loss))/x_train.shape[1]
    return cost
    

# 前后向传播过程
def fbp(w, b, x_train, y_train):
    # 前向传播
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # 后向传播过程
    dw = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    db = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"dw":dw, "db":db}
    return cost, gradients
    
    
# 更新参数
@change_dir
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # 更新(学习)
    for i in range(number_of_iteration):
        cost, gradients = fbp(w, b, x_train, y_train)
        cost_list.append(cost)
        # 更新
        w = w - learning_rate * gradients["dw"]
        b = b - learning_rate * gradients["db"]
        if i % 10 == 0:
            cost_list2 .append(cost)
            index.append(i)
            print("第%i次迭代后的成本:%f" % (i, cost))
        
    parameters = {"weight":w, "bias":b}
    plt.figure()
    plt.plot(index,  cost_list2)
    plt.savefig("./output/learning_curve.png")
    plt.close()
    return parameters, gradients, cost_list
    
    
# 进行预测
def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test)+b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
            
    return Y_prediction
    
    
# 初始化参数和层数
def init_nn_parameters(x_train, y_train):
    np.random.seed(42)
    parameters = {
        "weight1" : np.random.randn(3, x_train.shape[0])*0.1,
        "bias1" : np.zeros((3, 1)),
        "weight2" : np.random.randn(y_train.shape[0], 3)*0.1,
        "bias2" : np.zeros((y_train.shape[0], 1))
    }
    return parameters
    
    
# 神经网络前向传播过程
def fp_NN(x_train, parameters):
    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2
    }
    
    return A2, cache
    
    
# 神经网络的损失函数
def cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2), Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost
    
    
# 神经网络后向传播过程
def bp_NN(parameters, cache, X, Y):
    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T, dZ2)*(1-np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1, X.T)/X.shape[1]
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/X.shape[1]
    grads = {
        "dweight1" : dW1,
        "dbias1" : db1,
        "dweight2" : dW2,
        "dbias2" : db2
    }
    return grads
    
    
# 更新神经网络参数
def update_NN(parameters, grads, learning_rate = 0.01):
    parameters = {
        "weight1" : parameters["weight1"] - learning_rate*grads["dweight1"],
        "bias1" : parameters["bias1"] - learning_rate*grads["dbias1"],
        "weight2" : parameters["weight2"] - learning_rate*grads["dweight2"],
        "bias2" : parameters["bias2"] - learning_rate*grads["dbias2"]
    }
    return parameters
    
    
# 进行预测
def predict_NN(parameters, x_test):
    A2, cache = fp_NN(x_test, parameters)
    Y_pred = np.zeros((1, x_test.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_pred[0, i] = 0
        else:
            Y_pred[0, i] = 1
            
    return Y_pred
    
    
# 建立两层神经网络
@change_dir
def NN(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index_list = []
    # 初始化参数
    parameters = init_nn_parameters(x_train, y_train)
    print(parameters)
    for i in range(0, num_iterations):
        A2, cache = fp_NN(x_train, parameters)
        cost = cost_NN(A2, y_train, parameters)
        grads = bp_NN(parameters, cache, x_train, y_train)
        parameters = update_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print("第%i次迭代后的成本:%f" % (i, cost))
    
    plt.figure()
    plt.plot(index_list, cost_list)
    plt.savefig("./output/NN_LC.png")
    plt.close()
    
    # 进行预测
    y_pred_test = predict_NN(parameters, x_test)
    y_pred_train = predict_NN(parameters, x_train)
    # 计算准确率
    train_accuracy = 100 - np.mean(np.abs(y_pred_train - y_train))*100
    test_accuracy = 100 - np.mean(np.abs(y_pred_test - y_test))*100
    print("训练集预测准确率%f" % (train_accuracy))
    print("测试集预测准确率%f" % (test_accuracy))
    
    
# L层神经网络
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
    
def build_classifier():
    classifier = Sequential() # 初始化神经网络
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def L_NN(x_train, x_test, y_train, y_test):
    x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
    classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
    accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
    mean = accuracies.mean()
    variance = accuracies.std()
    print("平均准确度:", mean)
    print("准确度离散值:", variance)
    
    
@change_dir
def DP():
    x_train, y_train, x_test, y_test = loadData()
    w, b = init_params(x_train.shape[0])
    print(w, b)
    print(sigmoid(0))
    cost = fp(w, b, x_train, y_train)
    print(cost)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009, number_of_iteration = 200)
    print(parameters)
    # 进行预测
    y_pred_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_pred_train = predict(parameters["weight"], parameters["bias"], x_train)
    # 计算准确率
    train_accuracy = 100 - np.mean(np.abs(y_pred_train - y_train))*100
    test_accuracy = 100 - np.mean(np.abs(y_pred_test - y_test))*100
    print("训练集预测准确率%f" % (train_accuracy))
    print("测试集预测准确率%f" % (test_accuracy))
    
    # 用sklearn进行
    from sklearn import linear_model
    logreg = linear_model.LogisticRegression(random_state = 42, max_iter = 150)
    print("sklearn算法")
    print("训练集预测准确率%f" % (logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
    print("测试集预测准确率%f" % (logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
    
    # 二层神经网络
    parameters = NN(x_train, y_train, x_test, y_test, num_iterations = 2500)
    
    # L层神经网络
    # L_NN(x_train, x_test, y_train, y_test)
    
    
# pytorch学习
import numpy as np

# numpy数组
def numpy_array():
    array = [[1, 2, 3], [4, 5, 6]]
    first_array = np.array(array)
    print(type(first_array))
    print(np.shape(first_array))
    print(first_array)
    return first_array
    
    
import torch
# 张量
def pytorch_tensor(array):
    tensor = torch.Tensor(array)
    print(tensor.type)
    print(tensor.shape)
    print(tensor)
    
    
# 测试pytorch
def test_pytorch():
    array = numpy_array()
    pytorch_tensor(array)
    transform()
    basic_math()
    grad()
    linear_regress()
    logistic_regress2()
    ANN()
    
    
# 张量与数组的转换
def transform():
    array = np.random.rand(2, 2)
    print("{} {}\n".format(type(array), array))
    
    from_numpy_to_tensor = torch.from_numpy(array)
    print("{}\n".format(from_numpy_to_tensor))
    
    tensor = from_numpy_to_tensor
    from_tensor_to_numpy = tensor.numpy()
    print("{} {}\n".format(type(from_tensor_to_numpy), from_tensor_to_numpy))
    
    
from torch.autograd import Variable
# 求y = x^2 在x = [2, 4]的梯度
def grad():
    var = Variable(torch.ones(3), requires_grad = True)
    print(var)
    array = [2, 4]
    tensor = torch.Tensor(array)
    x = Variable(tensor, requires_grad = True)
    y = x**2
    print("y=", y)
    
    o = (1/2)*sum(y)
    print("o=", o)
    
    # 反向传播
    o.backward()
    
    print("梯度:", x.grad)
    
    
# 基础数学
def basic_math():
    # 创建tensor
    tensor = torch.ones(3, 3)
    print("\n", tensor)
    # 改变大小
    print("{}{}\n".format(tensor.view(9).shape, tensor.view(9)))
    # 加
    print("加:{}\n".format(torch.add(tensor, tensor)))
    # 减
    print("减:{}\n".format(tensor.sub(tensor)))
    # 乘
    print("乘:{}\n".format(torch.mul(tensor, tensor)))
    # 除
    print("除:{}\n".format(torch.div(tensor, tensor)))
    # 均值
    tensor = torch.Tensor([1, 2, 3, 4, 5])
    print("均值:{}".format(tensor.mean()))
    # 均值
    print("标准差:{}".format(tensor.std()))
    
    
# 线性回归的例子
import torch
from torch.autograd import Variable
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
@change_dir
def linear_regress():
    # 车价
    car_prices_array = [3, 4, 5, 6, 7, 8, 9]
    car_price_np = np.array(car_prices_array, dtype = np.float32)
    car_price_np = car_price_np.reshape(-1, 1)
    car_price_tensor = Variable(torch.from_numpy(car_price_np))
    # 车销量
    number_of_car_sell_array = [7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
    number_of_car_sell_np = np.array(number_of_car_sell_array, dtype = np.float32)
    number_of_car_sell_np = number_of_car_sell_np.reshape(-1, 1)
    number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))
    # 可视化
    plt.figure()
    plt.scatter(car_prices_array, number_of_car_sell_array)
    plt.savefig("./output/price_sell.png")
    plt.close()
    
    class LinearRegression(nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            return self.linear(x)
            
    input_dim = 1
    output_dim = 1
    model = LinearRegression(input_dim, output_dim)
    loss_fn = nn.MSELoss()
    
    # 优化器
    learning_rate = 0.02
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    # 训练模型
    loss_list = []
    iteration_number = 1001
    for iteration in range(iteration_number):
        # 优化
        optimizer.zero_grad()
        # 前向传播获得输出
        results = model(car_price_tensor)
        # 计算损失
        loss = loss_fn(results, number_of_car_sell_tensor)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 保存损失值
        loss_list.append(loss.data)
        # 打印损失值
        if iteration % 50 == 0:
            print("epoch {}, loss {}".format(iteration, loss.data))
            
    # 画图
    plt.figure()
    plt.plot(range(iteration_number), loss_list)
    plt.savefig("./output/lr_curve.png")
    
    # 进行预测
    predicted = model(car_price_tensor).data.numpy()
    plt.figure()
    plt.scatter(car_prices_array, number_of_car_sell_array, color = "red")
    plt.scatter(car_prices_array, predicted, color = "blue")
    plt.savefig("./output/result.png")
    plt.close()
    
    
# 逻辑回归的例子, 没调通，用另一个
@change_dir
def logistic_regress():
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print("逻辑回归\n")
    train = pd.read_csv("minst_train.csv", dtype = np.float32)
    print(train.info())
    print(train.describe())
    print(train.columns)
    
    # 提取特征和标签
    targets_numpy = train.label.values
    features_numpy = train.loc[:, train.columns != "label"].values/255 # 归一化
    print("np.array")
    print(len(targets_numpy))
    print(len(features_numpy))
    # 分割训练集和测试集
    features_train, features_test, targets_train, targets_test = train_test_split(targets_numpy, features_numpy, test_size = 0.2, random_state = 42)
    # 创建张量
    featuresTrain = torch.from_numpy(features_train)
    targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)
    
    # 定义超参数
    batch_size = 100
    n_iters = 10000
    num_epochs = n_iters / (len(features_train)/batch_size)
    num_epochs = int(num_epochs)
    
    # 创建数据集
    train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
    test = torch.utils.data.TensorDataset(featuresTest, targetsTest)
    print("tensor dataset")
    print(len(train[0]))
    print(len(test))
    
    # 数据加载器
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    
    # 数据可视化
    plt.figure()
    plt.imshow(features_numpy[10].reshape(28,28))
    plt.title(str(targets_numpy[10]))
    plt.savefig('./output/graph.png')
    plt.close()
    
    # 建立逻辑回归模型
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LogisticRegressionModel, self).__init__()
            # 线性部分
            self.linear = nn.Linear(input_dim, output_dim)
            # 接下来应该是逻辑斯蒂函数，
            # 但在pytorch中它在损失函数中了。
            
        def forward(self, x):
            print(type(x))
            out = self.linear(x)
            return out
            
            
    # 实例化模型
    input_dim = 28*28
    output_dim = 10
    model = LogisticRegressionModel(input_dim, output_dim)
    
    # 交叉熵损失函数
    error = nn.CrossEntropyLoss()
    # SGD优化器
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练模型
    count = 0
    loss_list = []
    iteration_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(images.size(), labels.size())
            # 定义变量
            # train = Variable(images.view(-1, 28*28))
            # labels = Variable(labels)
            # 清除梯度
            optimizer.zero_grad()
            # 前向过程
            outputs = model(train)
            # 计算softmax和交叉熵损失
            loss = error(outputs, labels)
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            
            count += 1
            
            # 预测
            if count % 50 == 0:
                # 计算准确率
                correct = 0
                total = 0
                # 预测测试集
                for images, labels in test_loader:
                    # test = Variable(images.view(-1, 28*28))
                    outputs = model(test)
                    # 用最大值作为预测值
                    predicted = torch.max(output.data, 1)[1]
                    # 标签总数
                    total += len(labels)
                    # 总的预测正确率
                    correct += (predicted == labels).sum()
                accuracy = 100 * correct / float(total)
                # 保存损失数据和迭代次数
                loss_list.append(loss.data)
                iteration_list.append(count)
            
            # 输出
            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
                
                
# 逻辑回归的例子
@change_dir
@timethis
def logistic_regress2():
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
    
    # 定义超参数
    input_size = 784    #输入层神经元大小
    num_classes = 10 #图像类别
    num_epochs = 25  #迭代次数
    batch_size = 100   #每次训练取得样本数
    learning_rate = 0.05 #学习率
    
    # 加载数据
    train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)
    
    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = nn.Linear(input_size, num_classes)
    
    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    # 训练模型
    loss_list = []
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 将数据变换为[每批大小, 图像大小]
            images = images.reshape(-1, 28*28)
            
            # 前向传播
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
    plt.figure()
    plt.plot(loss_list)
    plt.savefig("./output/lr_loss.png")
    plt.close()
                
    # 测试
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size()[0]
            correct += (pred == labels).sum()
            
        print("模型预测准确率{}%".format(100*correct.item()/total))
        
        
# 人工神经网络
@change_dir
@timethis
def ANN():
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
    
    print("ANN\n")
    # print(os.getcwd())
    # 加载数据
    train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), download=True)
    
    # 定义超参数
    batch_size = 100
    num_epochs = 100
    
    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    
    # 定义模型
    class ANNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(ANNModel, self).__init__()
            # 第一层 784->150
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            # 非线性成分
            self.relu1 = nn.ReLU()
            # 第二层 150-150
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            # 非线性成分
            self.tanh2 = nn.Tanh()
            # 第三层 150-150
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            # 非线性成分
            self.elu3 = nn.ELU()
            # 第四层 150-10
            self.fc4 = nn.Linear(hidden_dim, output_dim)
            
        # 前向传播
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.tanh2(out)
            out = self.fc3(out)
            out = self.elu3(out)
            out = self.fc4(out)
            
            return out
            
    # 初始化ANN
    input_dim = 28*28
    hidden_dim = 150 # 这个可以调参的
    output_dim = 10
    
    # 创建ANN
    model = ANNModel(input_dim, hidden_dim, output_dim)
    
    # 超参数
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.02
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 
    
    # 训练ANN
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # print(epoch, i)
            train = images.reshape(-1, 28*28)
            labels = Variable(labels)
            # 梯度清零
            optimizer.zero_grad()
            # 前向过程
            outputs = model(train)
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            
            count += 1
            if count % 50 == 0:
                # 计算准确率
                correct = 0
                total = 0
                for images, labels in test_loader:
                    test = images.reshape(-1, 28*28)
                    outputs = model(test)
                    pred = torch.max(outputs.data, 1)[1]
                    total += len(labels)
                    correct += (pred == labels).sum()
                    accuracy = 100*correct/float(total)
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            if count % 500 == 0:
                print('迭代次数: {}  损失: {}  准确率: {} %'.format(count, loss.data, accuracy))
    
    # 结果可视化
    plt.figure()
    plt.plot(iteration_list,loss_list)
    plt.title("ANN loss")
    plt.savefig("./output/ANN_loss.png")
    plt.figure()
    plt.plot(iteration_list,accuracy_list,color = "red")
    plt.title("ANN accuracy")
    plt.savefig("./output/ANN_accuracy.png")
    plt.close()
                           


if __name__ == "__main__":
    # DP()
    test_pytorch()