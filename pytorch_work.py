# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 实际自己工作的代码
# 用pytorch


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import janestreet

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import optuna
# 逻辑回归
from sklearn.linear_model import LinearRegression, LogisticRegression
# 支持向量机
from sklearn.svm import SVC, LinearSVC
# 随机森林
from sklearn.ensemble import RandomForestClassifier
# KNN算法
from sklearn.neighbors import KNeighborsClassifier
# 朴素贝叶斯算法
from sklearn.naive_bayes import GaussianNB
# SGD算法
from sklearn.linear_model import SGDClassifier
# 决策树算法
from sklearn.tree import DecisionTreeClassifier

import os

from EDA import data_explore
from FE import featureEngineer
from tools import *

    
    
# 建模前处理数据
def preprocessing(train):
    X_train = train.loc[:, train.columns.str.contains('feature')]
    # y_train = train.loc[:, 'resp']
    y_train = train.loc[:, 'action']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=666, test_size=0.2)
    
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
    
    
# 进行预测，生成提交文件，求值版
def predict_value(model):
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in iter_test:
        if test_df['weight'].item() > 0:
            test_df = featureEngineer(test_df)
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            # X_test = X_test.fillna(-999)
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
            test_df = featureEngineer(test_df)
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test = X_test.fillna(0.0)
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
    p = 0.01
    train = loadData(p = p)
    train = featureEngineer(train)
    # print(train.head())
    
    # 计算模型评分
    # score = Score(model, train)
    # print("模型评分:%.2f" % score)
    test = loadData(p = p)
    test = featureEngineer(test)
    
    #训练数据预处理
    X_train, y_train = preprocessing(train)
    
    # 逻辑回归
    print("逻辑回归")
    model = LogisticRegression(max_iter = 3000)
    model.fit(X_train, y_train)
    score(model, test, "Logist")
    
    # 支持向量机
    print("支持向量机")
    model = SVC()
    model.fit(X_train, y_train)
    score(model, test, "SVC")
    
    # 随机森林
    print("随机森林")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    score(model, test, "RandomForest")
    
    # knn
    print("knn")
    model = KNeighborsClassifier(n_neighbors = 2)
    model.fit(X_train, y_train)
    score(model, test, "knn")
    
    # 朴素贝叶斯
    print("朴素贝叶斯")
    model = GaussianNB()
    model.fit(X_train, y_train)
    score(model, test, "Bayes")
    
    # SGD算法
    print("SGD算法")
    model = SGDClassifier()
    model.fit(X_train, y_train)
    score(model, test, "SGD")
    
    # 决策树
    print("决策树算法")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    score(model, test, "DecisionTree")
    # 进行预测
    # predict_clf(model)
    