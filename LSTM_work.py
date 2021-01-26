# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 实际自己工作的代码
# LSTM模型


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import janestreet

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

import os

from FE import featureEngineer
from tools import *

    
    
# 建模前处理数据
def preprocessing(train):
    X_train = train.loc[:, train.columns.str.contains('feature')]
    # y_train = train.loc[:, 'resp']
    y_train = train.loc[:, 'action']
    
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=666, test_size=0.2)
    
    return X_train, y_train

    
# 评分函数
def Score(model, data):
    # test_df = pd.read_csv("/kaggle/input/jane-street-market-prediction/train.csv")
    data = data.fillna(-999)
    X_test = data.loc[:, data.columns.str.contains('feature')]
    resp = model.predict(X_test)
    date = data["date"].values
    weight = data["weight"].values
    action = (resp > 0).astype("int")
    
    count_i = len(np.unique(date))
    Pi = np.zeros(count_i)
    # 用循环太慢
    #for i, day in enumerate(np.unique(date)):
#        Pi[i] = np.sum(weight[date == day] * resp[date == day] * action[date == day])
    # 用下面这行代替
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u
    

# 进行预测，生成提交文件，分类版
def predict_clf(model):
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in iter_test:
        if test_df['weight'].item() > 0:
            # test_df = featureEngineer(test_df)
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test = X_test.fillna(0.0)
            y_preds = model.predict(X_test)[0]
        else:
            y_preds = 0
        # print(y_preds)
        sample_prediction_df.action = y_preds
        env.predict(sample_prediction_df)
        
        
class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size = 10, output_size = 1, num_layers = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
        
    def forward(self, _x):
        x, _ = self.lstm(_x)
        s, b, h = x.shape # seq_len, batch, hidden_size
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x
        

if __name__ == "__main__":
    newpath = "/home/code"
    os.chdir(newpath)
    
    # data_explore()
    
    # 真正开始干活
    p = 0.001
    train = loadData(p = p)
    train = featureEngineer(train)
    print(train.info())
    # print(train.head())
    
    # 计算模型评分
    # score = Score(model, train)
    # print("模型评分:%.2f" % score)
    test = loadData(p = p)
    test = featureEngineer(test)
    
    #训练数据预处理
    x_train, y_train = preprocessing(train)
    x_test, y_test = preprocessing(test)
    
    # 深度学习
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x_train.values.reshape(-1, 1, 130)
    # y_train.values.reshape(-1, 1, 1)
    x_tensor = torch.from_numpy(x_train.values.reshape(-1, 1, 130)).float().to(device)
    y_tensor = torch.from_numpy(y_train.values.reshape(-1, 1, 1)).float().to(device)
    

    Model = LstmRNN(130, 5).to(device)
            
    # model = Model(x_tensor).to(device)
    # print(model.state_dict())
    # 设置超参数
    lr = 0.000678
    n_epochs = 110
     
    # loss_fn = nn.BCELoss(reduction='sum')
    loss_fn = nn.MSELoss(reduction = "mean")
    optimizer = optim.Adam(Model.parameters(), lr = lr)
    # 创建训练器
    train_step = make_train_step(Model, loss_fn, optimizer)
    losses = []
    
    print("开始训练")
    # 进行训练
    for epoch in range(n_epochs):
        # y_tensor = y_tensor.detach()
        loss = train_step(x_tensor, y_tensor)
        losses.append(loss)
        
    # print(model.state_dict())
    print(losses)
    plt.figure()
    plt.plot(losses)
    plt.savefig("./output/loss.png")
    # 验证模型
    # x_test.reshape(-1, 1, 130)
    # y_test.reshape(-1, 1, 1)
    x_test_tensor = torch.from_numpy(x_test.values.reshape(-1, 1, 130)).float().to(device)
    y_test_tensor = torch.from_numpy(y_test.values.reshape(-1, 1, 1)).float().to(device)
    result = []
    for x in Model(x_test_tensor):
        if x >= 0.5:
            result.append(1)
        else:
            result.append(0)
    y_test = y_test_tensor.numpy()
    # print(len(y_test))
    # print(result)
    count = 0
    for i in range(len(result)):
        if y_test[i] == result[i]:
            count += 1
    print(count)
    print("预测正确率:%f" % (count/len(y_test)))
    # 进行预测
    # predict_clf(model)
    