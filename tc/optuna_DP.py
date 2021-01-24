# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 实际自己工作的代码
# 用optuna对深度学习模型调参


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
import optuna

import os

from FE import featureEngineer
from tools import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
# 建模前处理数据
def preprocessing(train):
    X = train.loc[:, train.columns.str.contains('feature')]
    # y_train = train.loc[:, 'resp']
    Y = train.loc[:, 'action']
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=666, test_size=0.2)
    
    return x_train, x_test, y_train, y_test 

    
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
            test_df = featureEngineer(test_df)
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test = X_test.fillna(0.0)
            y_preds = model.predict(X_test)[0]
        else:
            y_preds = 0
        # print(y_preds)
        sample_prediction_df.action = y_preds
        env.predict(sample_prediction_df)
        
        
# 获取数据
def getData():
    p = 0.1
    data = loadData(p = p)
    data = featureEngineer(data)
    # print(data.info())
    
    #训练数据预处理
    x_train, x_test, y_train, y_test  = preprocessing(data)
    
    return x_train, y_train, x_test, y_test
    
    
# 获取模型准确率
def getAccuracyRate(Model):
    result = []
    for x in Model(x_test_tensor):
        if x >= 0.5:
            result.append(1)
        else:
            result.append(0)
    y_test = y_test_tensor.numpy()
    # print(y_test[:10])
    # print(result[:10])
    count = 0
    for i in range(len(result)):
        if y_test[i] == result[i]:
            count += 1
    
    return count/len(y_test)
    
    
# 定义模型
def define_model(trial):
    input_dim = 130
    hide1_dim = trial.suggest_int("hide1_dim", 100, 200)
    hide2_dim = trial.suggest_int("hide2_dim", 10, 200)
    output_dim = 1
    Model = nn.Sequential(
            nn.Linear(input_dim, hide1_dim),
            nn.ReLU(),
            nn.Linear(hide1_dim, hide2_dim),
            nn.Sigmoid(),
            nn.Linear(hide2_dim, output_dim)
    )
    return Model
    
    
# 加载数据，为避免反复读取和数据一致，用全局变量
x_train, y_train, x_test, y_test = getData()
x_tensor = torch.from_numpy(x_train.values).float().to(device)
y_tensor = torch.from_numpy(y_train.values).float().to(device)
x_test_tensor = torch.from_numpy(x_test.values).float().to(device)
y_test_tensor = torch.from_numpy(y_test.values).float().to(device)
    
    
# 优化目标函数
@timethis
def objective(trial):
    Model = define_model(trial).to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(Model.parameters(), lr=lr)
    n_epochs = trial.suggest_int("epochs", 50, 200)
    loss_fn = nn.MSELoss(reduction = "mean")
    
    # 创建训练器
    train_step = make_train_step(Model, loss_fn, optimizer)
    # losses = []
    
    # 进行训练
    for epoch in range(n_epochs):
        # y_tensor = y_tensor.detach()
        loss = train_step(x_tensor, y_tensor)
        # losses.append(loss)
    accuracy = getAccuracyRate(Model)
    
    return accuracy
    

if __name__ == "__main__":
    newpath = "/home/code"
    os.chdir(newpath)
    
    # 用optuna进行调参
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("结果:", study.best_params)
    print(study.best_value)
    print(study.best_trial)
    
    # 进行预测
    # predict_clf(model)
    