# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 服务器版本


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib
import pickle
from run import *
import socket
import sys
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion
import janestreet


# 获取转换器
def getTransformer(X, y):
    transformer = FeatureUnion(
            transformer_list = [
                    ("features", SimpleImputer(strategy = "mean")),
                    ("indicators", MissingIndicator())
            ]
    )
    transformer = transformer.fit(X, y)
    return transformer


# 特征工程
def fp(data):
    print(data.info(verbose = True, null_counts = True))
    ds = data.describe()
    data = data[data["weight"] != 0]
    data["action"] =((data["weight"].values * data["resp"].values) > 0).astype("int")
    # 查看缺失值
    print(data.isnull().sum())
    # 复制数据，进行操作
    newdata = data.copy()
    # 特征列名称
    features = [c for c in newdata.columns if 'feature' in c] + ["date"]
    # 处理缺失值
    
    X = newdata.loc[:, features]
    y = newdata.loc[:, "action"]
    transformer = getTransformer(X, y)
#    X = transformer.transform(X)
    X = X.fillna(-999)
    print("特征工程结束")
    return (X, y, features, transformer)
    
    
# 形成提交文件
def makeSubmittion(model, features, transformer):
    print("正在生成提交文件")
    env = janestreet.make_env()
    iter_test = env.iter_test()
    
    for (test_df, pred_df) in iter_test:
        X_test = test_df.loc[:, features]
        #X_test = transformer.transform(X_test)
        X_test = X_test.fillna(-999)
        preds = model.predict(X_test)
        action = ((test_df['weight'].values * preds) > 0).astype('int')
        pred_df.action = action
        env.predict(pred_df)
    
    
# 线性回归模型
def LR(X, y):
    train_set, test_set, train_action, test_action = train_test_split(X, y, test_size = 0.2)
    print(len(train_set))
    # 训练
    linreg = LinearRegression()
    linreg.fit(train_set, train_action)
    # 预测
    train_pred = linreg.predict(train_set)
    test_pred = linreg.predict(test_set)
    # 模型评估
    print("train MSE:", metrics.mean_squared_error(train_action, train_pred))
    print("test MSE:", metrics.mean_squared_error(test_action, test_pred))
    print("train RMSE:", np.sqrt(metrics.mean_squared_error(train_action, train_pred)))
    print("test RMSE:", np.sqrt(metrics.mean_squared_error(test_action, test_pred)))
    # 保存模型到文件
    # joblib.dump(linreg, "LinesRegress.pkl")
    with open("/home/code/output/LinesRegress.pkl", "wb") as fw:
        pickle.dump(linreg, fw)
    print(test_pred)
    fig = plt.figure()
    plt.plot(test_pred[:100], "b.")
    plt.plot(test_action[:100], "rx")
    plt.savefig("/home/code/output/LR_result.png")
    return linreg
    
    
if __name__ == "__main__":
    print(os.getcwd())
    data = pd.read_csv("/home/code/small_train.csv", index_col = 0)
    X, y, features, transformer = fp(data)
    model = LR(X, y)
    makeSubmittion(model, features, transformer)
    