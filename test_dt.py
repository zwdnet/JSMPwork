# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 测试datatable的代码


import datatable as dt
import pandas as pd
from run import *


# 测试计时函数
@change_dir
@timethis
def testtime():
    print(3)
    sum = 0
    N = 1000
    for i in range(N):
        for j in range(N):
            sum += i*j
    print("sum = {}".format(sum))
    
    
# 读取数据
@change_dir
@timethis
def testread():
    train_df = dt.fread("./train.csv")
    print(train_df.shape)
    print(train_df.info())
    print(train_df.describe())
    print(train_df.sum())


if __name__ == "__main__":
    testread()
