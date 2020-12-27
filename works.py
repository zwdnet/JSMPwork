# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 实际自己工作的代码


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import janestreet

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import optuna
from sklearn.linear_model import LinearRegression, LogisticRegression

import os

from EDA import data_explore

    
    
# 建模过程
def modeling(train):
    print("开始建模")
    X_train = train.loc[:, train.columns.str.contains('feature')]
    # y_train = train.loc[:, 'resp']
    y_train = train.loc[:, 'action']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=666, test_size=0.2)
    # model = LinearRegression()
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

    
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


# 特征工程
def featureEngineer(data):
    data = data[data['weight'] != 0]
    data = data.fillna(-999)
    weight = data['weight'].values
    resp = data['resp'].values
    data['action'] = ((weight * resp) > 0).astype('int')
    return data
    
    
# 进行预测，生成提交文件，求值版
def predict_value(model):
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in iter_test:
        if test_df['weight'].item() > 0:
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test = X_test.fillna(-999)
            y_resp = model.predict(X_test)[0]
            y_preds = 0 if y_resp < 0 else 1
        else:
            y_preds = 0
        # print(y_preds)
        sample_prediction_df.action = y_preds
        env.predict(sample_prediction_df)
        
        
# 进行预测，生成提交文件，分类版
def predict_clf(model):
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in iter_test:
        if test_df['weight'].item() > 0:
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test = X_test.fillna(-999)
            y_preds = model.predict(X_test)[0]
        else:
            y_preds = 0
        # print(y_preds)
        sample_prediction_df.action = y_preds
        env.predict(sample_prediction_df)


if __name__ == "__main__":
    newpath = "/home/code"
    os.chdir(newpath)
    
    # data_explore()
    
    # 真正开始干活
    train = pd.read_csv("./train.csv", nrows = 10000)
    train = featureEngineer(train)
    model = modeling(train)
    # 计算模型评分
    # score = Score(model, train)
    # print("模型评分:%.2f" % score)
    
    # 进行预测
    predict_clf(model)
    