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
    y_train = train.loc[:, 'action']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=666, test_size=0.2)
    
    del train
    
    # X_train = X_train.fillna(-999)
    sampler = TPESampler(seed=666)
    tm = "auto"
    
    def create_model(trial):
        max_depth = trial.suggest_int("max_depth", 2, 12)
        n_estimators = trial.suggest_int("n_estimators", 2, 600)
        learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.99)
        subsample = trial.suggest_uniform('subsample', 0.0001, 1.0)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.0000001, 1)
        model = XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=666,
        tree_method=tm,
        silent = 1
        )
        
        return model
        
    def objective(trial):
        model = create_model(trial)
        model.fit(X_train, y_train)
        score = accuracy_score(
            y_train,
            model.predict(X_train)
            )
        return score
        
    params1 = {
        'max_depth': 8, 
        'n_estimators': 500, 
        'learning_rate': 0.01, 
        'subsample': 0.9, 
        'tree_method': tm,
        'random_state': 666
    }
    
    params3 = {
        'max_depth': 10, 
        'n_estimators': 500, 
        'learning_rate': 0.03, 
        'subsample': 0.9, 
        'colsample_bytree': 0.7,
        'tree_method': tm,
        'random_state': 666
    }
    
    start_time = time.time()
    model1 = XGBClassifier(**params1)
    model1.fit(X_train, y_train, eval_metric='auc')
    model1.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc',verbose=False)
    evals_result = model1.evals_result()
    print("模型1评分")
    y_true, y_pred = y_test, model1.predict(X_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
    
    model3 = XGBClassifier(**params3)
    model3.fit(X_train, y_train, eval_metric='auc')
    model3.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc',verbose=False)
    evals_result = model3.evals_result()
    print("模型3评分")
    y_true, y_pred = y_test, model3.predict(X_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_true, y_pred))
    end_time = time.time()
    print("建模时间:%.2f秒" % (end_time - start_time))
    
    return (model1, model3)


# 特征工程
def featureEngineer(data):
    data = data[data['weight'] != 0]
    data['action'] = ((data['weight'].values * data['resp'].values) > 0).astype('int')
    data = data.fillna(-999)
    return data


if __name__ == "__main__":
    newpath = "/home/code"
    os.chdir(newpath)
    # pio.orca.config.use_xvfb = True
    # pio.orca.config.executable = "/opt/conda/envs/tensorflow/bin/orca"
    pd.set_option('display.max_columns', None)
    
    # data_explore()
    
    # 真正开始干活
    train = pd.read_csv("./train.csv", nrows = 1000)
    train = featureEngineer(train)
    
    model1, model3 = modeling()
    
    # 进行预测
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in iter_test:
        if test_df['weight'].item() > 0:
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test = X_test.fillna(-999)
            y_preds = model1.predict(X_test) + model3.predict(X_test)
            if y_preds == 2:
                y_preds = np.array([1])
            else:
                y_preds = np.array([0])
        else:
            y_preds = np.array([0])
        sample_prediction_df.action = y_preds
        env.predict(sample_prediction_df)
    