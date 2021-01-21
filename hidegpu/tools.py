# coding:utf-8
# kaggle竞赛Jane Street Market Prediction
# 工具函数

from run import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import classification_report, roc_curve, auc


# 载入数据
@change_dir
def loadData(p = 0.01):
    # 抽样，读取1%数据
    # 参考https://mp.weixin.qq.com/s/2LSKnN9R-N-I2HcHePT9zA
    train = pd.read_csv("./train.csv", skiprows = lambda x: x>0 and np.random.rand() > p)
    # feature = pd.read_csv("./features.csv")
    return train
    
    
# 对模型进行交叉验证
def cross_val(model, X, Y, cv = 10):
    scores = cross_val_score(model, X, Y, cv=cv)
    score = scores.mean()
    return score
    
    
# 模型评估
def evalution(model, X, y_true):
    # X = test.loc[:, test.columns.str.contains("feature")].values
    # y_true = test.action.values
    y_pred = model.predict(X)
    target_names = ["1", "0"]
    result = classification_report(y_true, y_pred, target_names = target_names, output_dict = False )
    return result


# 对模型评分
@timethis
def score(model, test, modelName):
    if modelName == "XGBoost":
        X = test.loc[:, test.columns.str.contains("feature")]
        Y = test.action
    else:
        X = test.loc[:, test.columns.str.contains("feature")].values
        Y = test.action.values
    model_score = model.score(X, Y)
    cross_score = cross_val(model, X, Y)
    report = evalution(model, X, Y)
    print("模型评分:", model_score)
    print("交叉验证:", cross_score)
    print("模型评估:\n", report)
    Roc(model, X, Y, modelName)
    Lc(model, modelName, X, Y)
    
    
# 画roc曲线
@change_dir
def Roc(model, X, Y, modelName):
    y_label = Y
    y_pred = model.predict(X)
    fpr, tpr, thersholds = roc_curve(y_label, y_pred)
        
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'k--', label = "ROC (area = {0:.2f})".format(roc_auc), lw = 2)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(modelName + " ROC Curve")
    plt.legend(loc = "best")
    plt.savefig("./output/" + modelName + "_ROC.png")
    
    
# 画学习曲线
@change_dir
def Lc(model, modelName, X, y, ylim = None, cv = None, n_jobs = 1, train_sizes = np.linspace(0.1, 1.0, 5), verbose = 0):
    plt.figure()
    plt.title(modelName+" Learning Curve")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Samples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.savefig("./output/" + modelName + "_Learning Curve.png")
    
    
# 工具函数，返回神经网络训练的每一步
def make_train_step(model, loss_fn, optimizer):
    # 执行在循环中训练过程
    def train_step(x, y):
        # 设置训练模式
        model.train()
        # 预测
        yhat = model(x)
        # 计算损失
        # print("测试")
        yhat = yhat.squeeze(-1)
        # print(yhat.shape, y.shape)
        loss = loss_fn(yhat, y)
        # 计算梯度
        loss.backward()
        # 更新参数，梯度置零
        optimizer.step()
        optimizer.zero_grad()
        # 返回损失值
        return loss.item()
        
    # 返回在训练循环中调用的函数
    return train_step
        

    