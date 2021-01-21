# coding:utf-8
# kaggle Jane Street Market Prediction代码
# optuna的测试代码

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import os
# from tools import *
from FE import featureEngineer

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler

# XGBoost
from xgboost import XGBClassifier


def objective(trial):
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2)**2
    
    
def objective2(trial, x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, random_state = 101)
    param = {
        "eval_metric":trial.suggest_categorical("eval_metric", ["logloss"]),
        "tree_method":trial.suggest_categorical("tree_method", ["gpu_hist"]),
        "n_estimators" : trial.suggest_int('n_estimators', 1, 100),
        'max_depth':trial.suggest_int('max_depth', 2, 12),
        'learning_rate':trial.suggest_loguniform('learning_rate',0.001,0.5),
        "subsample":trial.suggest_loguniform("subsample", 0.5, 1.0)
    }
    model = XGBClassifier(**param)
    model.fit(train_x, train_y)
    
    return cross_val_score(model,test_x,test_y).mean()
    
    
# 建模前处理数据
def preprocessing(train):
    X_train = train.loc[:, train.columns.str.contains('feature')]
    # y_train = train.loc[:, 'resp']
    y_train = train.loc[:, 'action']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=666, test_size=0.2)
    
    return X_train, y_train


if __name__ == "__main__":
#    study = optuna.create_study()
#    study.optimize(objective, n_trials = 100)
#    print("结果:", study.best_params)
#    print(study.best_value)
#    print(study.best_trial)
#    study.optimize(objective, n_trials = 100)
#    print("结果:", study.best_params)
#    print(study.best_value)
#    print(study.best_trial)
    
    
    # data_explore()
    
    # 真正开始干活
    p = 0.001
    train = pd.read_csv("small_train.csv")
    train = featureEngineer(train)
    # print(train.head())
    
    # 计算模型评分
    # score = Score(model, train)
    # print("模型评分:%.2f" % score)
    
    #训练数据预处理
    X_train, y_train = preprocessing(train)
    
    # xgboost
    print("XGBoost")
    study = optuna.create_study(direction = "maximize", sampler = TPESampler())
    study.optimize(lambda trial:objective2(trial, X_train, y_train), n_trials = 100)
    print("结果:", study.best_params)
    print(study.best_value)
    print(study.best_trial)
    
