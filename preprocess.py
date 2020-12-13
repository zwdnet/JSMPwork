# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 数据探索及预处理


import pandas as pd
from run import *
import matplotlib.pyplot as plt
import dask.dataframe as dd


# 初步探索花了一天
@change_dir
def drawData():
    # data = pd.read_csv("train.csv", usecols = [0,1])
    n = 2390491
    row_read = int(n/100)
    # row_read = 5
    # data = pd.read_csv("./train.csv", nrows = row_read)
    data = dd.read_csv("./train.csv")
    # print(data.head())
    # print(data.info())
    print(data.info())
    print(data.columns)
    
    fig = plt.figure()
    plt.plot(data["weight"].values.compute())
    plt.savefig("./output/weight.png")
    
    s = "resp_"
    for i in range(1, 5):
        col = s+str(i)
        plt.close()
        fig = plt.figure()
        plt.plot(data[col].values.compute())
        plt.savefig("./output/"+col+".png")
        
    plt.close()
    fig = plt.figure()
    plt.plot(data["resp"].values.compute())
    plt.savefig("./output/"+"resp"+".png")
    
    s = "feature_"
    for i in range(0, 130):
        col = s+str(i)
        plt.close()
        fig = plt.figure()
        plt.plot(data[col].values.compute())
        plt.savefig("./output/"+col+".png")

    return data
    
    
# 读取数据，提取前1/10做研究
@change_dir
def smallData():
    n = 2390491
    row_read = int(n/10)
    data = pd.read_csv("./train.csv", nrows = row_read)
    print(data.info())
    # 画图
    fig = plt.figure()
    plt.plot(data["weight"].values)
    plt.savefig("./output/weight_small.png")
    
    s = "resp_"
    for i in range(1, 5):
        col = s+str(i)
        plt.close()
        fig = plt.figure()
        plt.plot(data[col].values)
        plt.savefig("./output/"+col+"_small.png")
        
    plt.close()
    fig = plt.figure()
    plt.plot(data["resp"].values)
    plt.savefig("./output/"+"resp"+"_small.png")
    
    s = "feature_"
    for i in range(0, 130):
        col = s+str(i)
        plt.close()
        fig = plt.figure()
        plt.plot(data[col].values)
        plt.savefig("./output/"+col+"_small.png")
    data.to_csv("./small_train.csv")
        
    
if __name__ == "__main__":
    data = pd.read_csv("small_train.csv")
    print(data.info())
    