# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 实际自己工作的代码


import numpy as np
import pandas as pd
import janestreet

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import LinearRegression

import os
import time


# 数据探索
def data_explore():
    # 读取数据
    train = pd.read_csv("./train.csv", nrows = 10000)
    print(train.head())
    
    # 先画图看目标特征的分布
    # .plt.figure()
    plot_list = ['weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']
    fig = make_subplots(rows=3, cols=2)
    traces = [
        go.Histogram(
            x = train[col],
            nbinsx = 100,
            name = col
        ) for col in plot_list
    ]
    
    for i in range(len(traces)):
        fig.append_trace(
            traces[i],
            (i // 2) + 1,
            (i % 2) + 1
        )
    
    fig.update_layout(
        title_text='Target features distributions',
        height = 900,
        width = 800
    )
    
    pio.write_image(fig, "./output/target_distribute.png")
    
    # 看特征值的分布
    features = train.columns
    features = features[7:]
    features = features[:130]
    fig = make_subplots(
        rows = 44,
        cols = 3
    )
    traces = [
        go.Histogram(
            x = train[col],
            nbinsx = 100,
            name = col
        ) for col in features
    ]
    
    for i in range(len(traces)):
        fig.append_trace(
            traces[i],
            (i // 3) + 1,
            (i % 3) + 1
        )
    
    fig.update_layout(
        title_text='Train features distributions',
        height = 5000
    )
    
    pio.write_image(fig, "./output/features_distribute.png")
    
    cols = features
    
    # 读取其它数据文件看看
    features = pd.read_csv("./features.csv")
    print(features)
    example_test = pd.read_csv("./example_test.csv")
    print(example_test)
    submission = pd.read_csv("./example_sample_submission.csv")
    print(submission)
    
    # 开始建模
    train = pd.read_csv("./small_train.csv")
    # 先找到高度相关的特征
    all_columns = []
    for i in range(0, len(cols)):
        for j in range(i+1, len(cols)):
            if abs(train[cols[i]].corr(train[cols[j]])) > 0.95:
                all_columns = all_columns + [cols[i], cols[j]]
    
    all_columns = list(set(all_columns))
    print('Number of columns:', len(all_columns))
    # 画图
    data = train[all_columns]
    f = plt.figure(
        figsize = (22, 22)
    )
    plt.matshow(
        data.corr(),
        fignum = f.number
    )
    plt.xticks(
        range(data.shape[1]),
        data.columns,
        fontsize = 14,
        rotation = 90
    )
    plt.yticks(
        range(data.shape[1]),
        data.columns,
        fontsize = 14
    )
    cb = plt.colorbar()
    cb.ax.tick_params(
        labelsize = 14
    )
    plt.savefig("./output/features_corr.png")
    
    # 目标值的相关度
    data = train[['weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']]
    f = plt.figure(
        figsize = (12, 12)
    )
    plt.matshow(
        data.corr(),
        fignum = f.number
    )
    plt.xticks(
        range(data.shape[1]),
        data.columns,
        fontsize = 14,
        rotation = 90
    )
    plt.yticks(
        range(data.shape[1]),
        data.columns,
        fontsize = 14
    )
    cb = plt.colorbar()
    cb.ax.tick_params(
        labelsize = 14
    )
    plt.savefig("./output/targets_corr.png")
    
    
# 建模过程
def modeling(train):
    print("开始建模")
    
    X_train = train.loc[:, train.columns.str.contains('feature')]
    y_train = train.loc[:, 'resp']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=666, test_size=0.2)
    model = LinearRegression()
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


if __name__ == "__main__":
    newpath = "/home/code"
    os.chdir(newpath)
    # pio.orca.config.use_xvfb = True
    # pio.orca.config.executable = "/opt/conda/envs/tensorflow/bin/orca"
    pd.set_option('display.max_columns', None)
    
    # data_explore()
    
    # 真正开始干活
    train = pd.read_csv("./train.csv", nrows = 10000)
    train = featureEngineer(train)
    model = modeling(train)
    # 计算模型评分
    score = Score(model, train)
    print("模型评分:%.2f" % score)
    
    # 进行预测
    predict_value(model)
    