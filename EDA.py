# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 数据探索代码


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os
import sys
import gc
from run import *
from sklearn.preprocessing import StandardScaler as scale
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# 数据探索
def data_explore_old(df):
    # 复制数据，防止改变原数据
    data = df.copy()
    # 查看列名
    print(data.columns)
    # 查看数据开头
    print(data.head())
    # 有空值
    # 看数据总和
    print(data.sum())
    # 看平均数
    print(data.mean())
    # 输出描述统计值
    print(data.describe())
    # 查看空值
    print(data.isnull().sum())
    # 最多的有1734个空值，接近20%
    # 先画折线图吧
    # 画目标值
    fig = plt.figure()
    fig, axes = plt.subplots(4, 2, sharex = True)
    for i in range(4):
        for j in range(2):
            pos = 2*i + j
            if pos > 6:
                break
            axes[i][j].set_title(data.columns[pos])
            axes[i][j].plot(data.iloc[:, pos])
    plt.subplots_adjust(wspace = 0.2, hspace = 1)
    plt.savefig("./output/targets_line.png")
    plt.close()
    # 画特征
    fig = plt.figure(figsize = (10, 80))
    for i in range(130):
        ax = fig.add_subplot(65, 2, i+1)
        ax.set_title(data.columns[i+7])
        plt.plot(data.iloc[:, i+7])
    plt.subplots_adjust(wspace = 0.2, hspace = 1)
    plt.savefig("./output/features_line.png")
    plt.close()
    
    # 画柱状图
    # 画目标值
    fig = plt.figure()
    sns.distplot(data.iloc[:, 1:8], hist = True, bins = 100, kde = True)
    # data.iloc[:, 1:8].plot.hist(subplots = True, sharex = True, layout = (4, 2), bins = 50)
    plt.savefig("./output/targets_hist.png")
    # 画特征
    fig = plt.figure()
    sns.distplot(data.iloc[:, 8:-2], hist = True, bins = 100, kde = True)
    # data.iloc[:, 8:-2].plot.hist(subplots = True, sharex = True, layout = (65, 2), figsize = (10, 80), bins = 50)
    plt.savefig("./output/features_hist.png")
    
#    # 画密度图
#    # 画目标值
#    fig = plt.figure()
#    data.iloc[:, 1:8].plot(subplots = True, kind = "hist", sharex = True, layout = (4, 2), bins = 50)
#    plt.savefig("./output/targets_hist.png")
#    # 画特征
#    fig = plt.figure()
#    data.iloc[:, 8:-2].plot(subplots = True, kind = "hist", sharex = True, layout = (65, 2), figsize = (10, 80), bins = 50)
#    plt.savefig("./output/features_hist.png")


# 数据探索
@change_dir
def data_explore():
    sns.set_style('darkgrid')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # 抽样，读取1%数据
    # 参考https://mp.weixin.qq.com/s/2LSKnN9R-N-I2HcHePT9zA
    train_df = pd.read_csv("./train.csv", skiprows = lambda x: x>0 and np.random.rand() > 0.01)
    test_df = pd.read_csv("./example_test.csv")
    feature_df = pd.read_csv("./features.csv")
    

    # 复制数据
    # train = train_df.copy()
    # test = test_df.copy()
    EDA1(train_df, test_df, feature_df)
    EDA2(train_df, test_df, feature_df)
    EDA3(train_df, test_df, feature_df)
    
    
# 第一篇文章的EDA
# 参考https://www.kaggle.com/muhammadmelsherbini/jane-street-extensive-eda
def EDA1(train, test, feature):
    df = train.copy()
    # 看数据长度
    org_len = len(df)
    print(org_len)
    # 查看数据概况
    print(df.info())

    # 按日期排序数据
    df.sort_values(by = ["date", "ts_id"], inplace = True)
    
    # 增加目标数据
    df["action"] = np.where(df["resp"] > 0, 1, 0)
    df.action = df.action.astype("category")
    
    # 下面开始分析数据
    # 先分析resp
    fig = plt.figure(figsize = (16, 6))
    ax = plt.subplot(1, 1, 1)
    df.groupby("date")[["resp_1", "resp_2", "resp_3", "resp_4", "resp"]].sum().cumsum().plot(ax = ax)
    plt.savefig("./output/01.png")
    # 前92天收益较高，resp_4的累积收益较高
    # resp_1的累积收益较低
    
    # 再画resp的平均值
    fig = px.line(df.groupby("date")[["resp_1", "resp_2", "resp_3", "resp_4", "resp"]].mean(), x = df.groupby("date")[["resp_1", "resp_2", "resp_3", "resp_4", "resp"]].mean().index, y = ["resp_1", "resp_2", "resp_3", "resp_4", "resp"], title = "average resp per day")
    fig.write_image("./output/02.png")
    
    # 画组图
    # 画resp数据的直方组图
    def resp_hists(ax1, ax2, ax3, data, name):
        ax1.hist(data, bins = 150, color = "darkblue", alpha = 0.6)
        ax1.axvline(data.mean() + data.std(),color = 'darkorange', linestyle = ':',linewidth = 2)
        ax1.axvline(data.mean() - data.std(),color = 'darkorange', linestyle = ':',linewidth = 2)
        data.plot.hist(bins = 150,ax = ax2, color = 'darkblue', alpha = 0.6)
        ax2.axvline(data.mean() + data.std(),color = 'darkorange', linestyle = ':', linewidth = 2)
        ax2.axvline(data.mean() - data.std(), color = 'darkorange', linestyle = ':',linewidth = 2)
        ax2.set_xlim(-.08, .08)
        ax3.hist(data, bins=150, color='darkblue',alpha=.6)
        ax3.set_yscale('log')
        skew= round(data.skew(),4)
        kurt= round(data.kurtosis())
        std1= round((((data.mean()-data.std()) < data ) & (data < (data.mean()+data.std()))).mean()*100,2)
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax1.text(.02,.96,'μ = {}\nstd = {}\nskewness = {}\nkurtosis = {}\n% values in 1 std = {}%'.format(round(data.mean(),4),round(data.std(),4),skew,kurt,std1),
transform=ax1.transAxes, verticalalignment='top',bbox=props,fontsize=10)
        ax1.set_title(name + ' Hist Normal scale', fontsize=14)
        ax2.set_title(name + ' Hist normal scale zoomed',fontsize=14)
        ax3.set_title(name + ' Hist with freq on a log scale',fontsize=14);
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax3.set_xlabel('')
        ax3.set_ylabel('')
    
    fig,((ax11,ax12,ax13),(ax21,ax22,ax23),(ax31,ax32,ax33),(ax41,ax42,ax43),(ax51,ax52,ax53)) = plt.subplots(5,3,figsize=(18,24))
    plt.subplots_adjust(hspace = 0.35)
    resp_hists(ax11, ax12, ax13, df.resp, "Resp")
    resp_hists(ax21, ax22, ax23, df.resp_1, "Resp_1")
    resp_hists(ax31, ax32, ax33, df.resp_2, "Resp_2")
    resp_hists(ax41, ax42, ax43, df.resp_3, "Resp_3")
    resp_hists(ax51, ax52, ax53, df.resp_4, "Resp_4")
    
    plt.savefig("./output/03.png")
    
    # resp变量之间配对作图
    sns.pairplot(df[["resp_1", "resp_2", "resp_3", "resp_4", "resp"]], corner = False)
    plt.savefig("./output/04.png")
    # resp与resp_4，以及resp_1与resp_2之间高度相关。
    # 投资时区越长，风险及收益越大，反之越小
    
    # 下面分析date
    # 看独特的date值
    print(df.date.unique())
    # 完整数据500天，大约两年的数据
    # 现在查看每天的收益总数，以及操作总数
    fig = px.area(data_frame = df.groupby("date")[["resp"]].count(), title='Number of operation per day')
    fig.update_traces(showlegend = False)

    fig.layout.xaxis.title = 'Day'
    fig.layout.yaxis.title = 'Number of operations'
    fig.write_image("./output/05.png")
    # 每天收益总数
    fig = px.area(data_frame = df.groupby("date")[["resp"]].sum(), title='Resp sum of operation per day')
    fig.update_traces(showlegend = False)

    fig.layout.xaxis.title = 'Day'
    fig.layout.yaxis.title = 'Resp sum of operations'
    fig.write_image("./output/06.png")
    # 可以看到收益有很多波动
    # 下面建立平均收益的20天移动标准差
    date_df = df.groupby("date")[["resp"]].mean()
    std20 = []
    for i in range(len(date_df)):
        if i < 20:
            std20.append(np.nan)
        else:
            moving_std = date_df["resp"][i-20:i].std()
            std20.append(moving_std)
    date_df["moving_std"] = std20
    print(date_df.tail(2))
    # 画图看看
    fig = px.line(data_frame = date_df, y = ["resp", "moving_std"], title='Average Resp & 20 day moving standard deviation')
    fig.layout.xaxis.title = "Day"
    fig.layout.yaxis.title = "Avg Resp"
    fig.write_image("./output/07.png")
    
    # 现在看每个resp值每天的标准差
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (14, 12))
    df.groupby("date")[["resp_1", "resp_2", "resp_3", "resp_4"]].std().plot(ax = ax1, color=['steelblue','darkorange','red','green'], alpha=.8)
    df.groupby("date")[["resp_1", "resp_2", "resp_3", "resp_4"]].std().plot.kde(ax = ax2)
    fig.suptitle('Resp\'s Std',fontsize=18,y=.96)

    ax2.set_xlabel('')

    ax1.set_xlabel('')

    ax2.set_title('kde of each resp std', fontsize=14)

    ax1.set_title('std of Resp\'s for each trading day',fontsize=14)
    fig.savefig("./output/08.png")
    # 更长时期的resp的标准差也更大
    # 另外前100天的标准差大一些，因为80天后模型有调整
    
    # 下面来看看weight的情况
    fig = plt.figure(figsize = (18, 7))
    grid = gridspec.GridSpec(2, 3, figure = fig, hspace = 3, wspace = 2)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[:, 2])
    sns.boxplot(x = df.weight, width = 0.5, ax = ax1)
    ax2.hist(df.weight, color='#404788ff', alpha = 0.6, bins = list([-.05] + list(10**np.arange(-2,2.24,.05))))
    ax2.set_xscale('symlog')

    ax2.set_xlim(-.05,227)
    sns.boxplot(x = df.weight[df.weight != 0], width = 0.5, ax = ax3)
    ax1.set_title('Weights including zero weights',fontsize=14)

    ax3.set_title('Weights not including zero weights',fontsize=14)

    ax2.set_title('Weights including zero weights (log)',fontsize=14)

    ax4.set_title('Weights not including zero weights (log)',fontsize=14)
    props = dict(boxstyle='round',facecolor='white', alpha=0.4)
    ax1.text(.2,.9,'μ = {} std = {}\nmin = {} max = {}'.format(round(df.weight.mean(),3),round(df.weight.std(),3),round(df.weight.min(),3), round(df.weight.max(),3)),
transform=ax1.transAxes, verticalalignment='top',bbox=props,fontsize=12)

    ax3.text(.2,.9,'μ = {} std = {}\nmin = {} max = {}'.format(round(df.weight[df.weight
!= 0].mean(),3), round(df.weight[df.weight != 0].std(), 3),
round(df.weight[df.weight != 0].min(),3), round(df.weight[df.weight != 0].max(),3)), transform=ax3.transAxes, verticalalignment='top',bbox=props, fontsize=12)
    ax4.hist(df.weight[df.weight !=0], color='#404788ff', alpha=.6,bins=10**np.arange(-2.16,2.24,.05))
    ax4.set_xscale('log')

    ax4.set_xticks((.01,.03,.1,.3,1,3,10,30,100))

    ax4.set_xticklabels((.01,.03,.1,.3,1,3,10,30,100))
    ax5.pie(((df.weight==0).mean(),(1-(df.weight==0).mean())),startangle=300,wedgeprops=dict(width=0.5), labels=('Zeros\n{}%'.format(round((df.weight==0).mean()*100,2)), 'Nonzeros\n{}%'.format(round((1-(df.weight==0).mean())*100,2))),
textprops={'fontsize': 12},colors=['#404788ff','#55c667ff'])
    ax5.set_title('Zeros vs non-zero weights', fontsize=14)

    ax1.set_xlabel('')

    ax2.set_xlabel('')

    ax3.set_xlabel('')

    ax2.set_ylabel('')

    ax5.set_ylabel('')

    ax4.set_xlabel('')
    fig.savefig("./output/09.png")
    # 画weight的直方图
    fig = plt.figure(figsize = (15, 10))
    fig.suptitle('Nonzero weights histogram in different scales',fontsize=18)
    ax1 = plt.subplot(3,1,1)
    ax1.hist(df.weight[df.weight !=0], color='darkblue', alpha=.7, bins=10**np.arange(-2.16,2.23,.05))
    plt.xscale('log')
    plt.xticks((.01,.03,.1,.3,1,3,10,30,100),(.01,.03,.1,.3,1,3,10,30,100))
    ax2 = plt.subplot(3, 1, 2)
    sns.distplot(df.weight[df.weight != 0], color='darkblue', bins=400, ax=ax2)
    ax3 = plt.subplot(3, 1, 3)
    ax3.hist(df.weight[(df.weight !=0) & (df.weight < 3.197 )],color='darkblue',alpha=.7, bins=200)
    ax3.set_xlim(0,3.3)

    ax2.set_xlabel('')

    ax1.set_title('All values (log-scale)', fontsize=14)

    ax2.set_title('kde of the distribution',fontsize=14)

    ax3.set_title('75% of the Values',fontsize=14)

    plt.subplots_adjust(hspace=.4)
    fig.savefig("./output/10.png")
    # 再看细一点
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (16, 8))
    fig.suptitle('Weight outliers',fontsize=18)
    sns.boxplot(df.weight, width = 0.5, ax = ax1)
    ax1.axvline(np.percentile(df.weight,95), color= 'green',label='95.0%',linestyle=':', linewidth=3)

    ax1.axvline(np.percentile(df.weight,99), color= 'darkblue',label='99.0%',linestyle=':',linewidth
= 3)

    ax1.axvline(np.percentile(df.weight,99.9), color= 'darkorange',label='99.9%',linestyle=':',linewidth=3)

    ax1.axvline(np.percentile(df.weight,99.99), color= 'magenta',label='99.99%',linestyle=':',linewidth=3)

    ax1.legend(fontsize=13)
    sns.boxplot(df.weight[df.weight != 0], width = 0.5, ax = ax2)
    ax2.axvline(np.percentile(df.weight[df.weight != 0],95), color= 'green',label='95.0%',linestyle=':', linewidth=3)

    ax2.axvline(np.percentile(df.weight[df.weight != 0],99), color= 'darkblue',label='99.0%',linestyle=':',linewidth
= 3)

    ax2.axvline(np.percentile(df.weight[df.weight != 0],99.9), color= 'darkorange',label='99.9%',linestyle=':',linewidth=3)

    ax2.axvline(np.percentile(df.weight[df.weight != 0],99.99), color= 'magenta',label='99.99%',linestyle=':',linewidth=3)

    ax2.legend(fontsize=13)
    fig.savefig("./output/11.png")
    
    # resp与weight的关系
    fig = plt.figure()
    sns.scatterplot(data = df, x = "resp", y = "weight", color = "blue", alpha = 0.3)
    plt.title('Resp vs Weight\ncorrelation={}'.format(round(df.weight.corr(df.resp),4)))
    plt.savefig("./output/12.png")
    # 两者不是线性相关的，高权重值与低收益值相关    # 再来看看feature数据。
    df_f = pd.read_csv("./features.csv")
    print(df_f.head(5))
    print(df_f.shape)
    # 画柱状图看看
    fig = px.bar(df_f.set_index("feature").T.sum(), title = "Number of tags for each feature")
    fig.layout.xaxis.tickangle = 300
    fig.update_traces(showlegend = False)
    fig.layout.xaxis.dtick = 5

    fig.layout.xaxis.title = ""
    fig.layout.yaxis.title = ""
    fig.write_image("./output/13.png")
    
    # 下面画图看空值
    fig = px.bar(x = df.isnull().sum().index, y = df.isnull().sum().values, title = "Number of null values")
    fig.layout.xaxis.tickangle = 300

    fig.layout.xaxis.dtick = 5

    fig.layout.yaxis.dtick = 100000

    fig.layout.xaxis.title = ''

    fig.layout.yaxis.title = ''

    fig.layout.xaxis.showgrid = True
    fig.write_image("./output/14.png")
    # 有空值的特征约占10%
    nulls = df.isnull().sum()
    nulls_list = list(nulls[nulls > (0.1*len(df))].index)
    print(nulls_list)
    # 看看这些特征中空值的个数有没有什么模式
    plt.figure()
    corr_null = df[['resp','resp_1','resp_2','resp_3','resp_4','weight']+nulls_list].corr()
    #print(corr_null)
    sns.heatmap(corr_null)
    plt.savefig("./output/15.png")
    # 将所有空值数量大于10%的特征丢弃
    df.drop(columns = nulls_list, inplace = True)
    # 现在关注剩下的空值，先看相关系数的变异
    print((df.iloc[:, 7:-2].std()/df.iloc[:7:-2].mean()).head(5))
    # 看起来相关性是不可靠的，因为均值接近0
    
    # 现在看特征值的分布
    plt.figure()
    df.iloc[:, 7:-2].hist(bins=100, figsize=(20, 74), layout=(29, 4))
    plt.savefig("./output/16.png")
    # 接着用水平箱图来看，由于数据较集中，排除末端0.1%
    fig = plt.figure(figsize=(20, 80))
    fig.suptitle('Features Box plot with 0.1% 99.9% whiskers',fontsize=22, y=.89)
    grid = gridspec.GridSpec(29,4,figure=fig,hspace=.5,wspace=.05)
    featstr = [i for i in df.columns[7:-2]]
    counter = 0
    for i in range(29):
        for j in range(4):
            subf = fig.add_subplot(grid[i, j])
            sns.boxplot(x = df[featstr[counter]], saturation = 0.5, color = "blue", ax = subf, width = 0.5, whis = (0.1, 99.9))
            subf.axvline(df[featstr[counter]].mean(),color= 'darkorange', label='Mean', linestyle=':',linewidth=3)
            subf.set_xlabel("")
            subf.set_title('{}'.format(featstr[counter]),fontsize=16)
            counter += 1
            # gc.collect()
    plt.savefig("./output/17.png")
    # 我们可以看到有很多异常值，影响特征分布
    # 由于大部分数据集中于均值附近，用均值填充空值
    df.fillna(df.mean(axis = 0), inplace = True)
    
    # 再来看看特征的累积情况
    df.groupby("date")[featstr].mean().cumsum().plot(layout=(29, 4), subplots = True, figsize=(20, 82), xlabel="")
    fig = plt.gcf()
    fig.text(0.5, 0.19, "Date", ha="center", fontsize=24)
    fig.suptitle("Cumulative features means per day", fontsize=24, y=0.886)
    fig.savefig("./output/18.png")
    # 大部分特征的累积均数是递增的，也有部分是递减，还有小部分是没有明显趋势的
    
    # 下面来看特征之间的相关度
    corr = df.iloc[:, 7:-2].corr()
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(corr, ax = ax, cmap = "coolwarm")
    plt.savefig("./output/19.png")
    # 看起来很多特征之间存在共线性
    featstr2 = [i for i in featstr if i not in ["feature_41", "feature_64"]]
    print(len(featstr))
    
    # 画散点图看看
    fig = plt.figure(figsize=(22, 44))
    grid = gridspec.GridSpec(12, 5, figure=fig, hspace=0.5, wspace=0.2)
    counter = 1
    for i in range(12):
        for j in range(5):
            if counter == 113:
                break
            subf = fig.add_subplot(grid[i, j])
            sns.scatterplot(x = df[featstr2[counter]], y = df[featstr2[counter+1]], ax = subf)
            cor = round(df[featstr2[counter]].corr(df[featstr2[counter+1]])*100, 2)
            subf.set_xlabel("")
            subf.set_ylabel("")
            subf.set_title('{} & {}\nCorrelation = {}%'.format(featstr2[counter],featstr2[counter+1],cor),fontsize=14)
            counter += 2
            # gc.collect()
    fig.savefig("./output/20.png")
    # 由于这些特征是金融相关特征，很多都有很高的相关性
    # 现在来看看有高度相关性的特征组
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[featstr2[15:23]].corr(), center = 0, cmap = "coolwarm", annot = True, cbar = False)
    fig.savefig("./output/21.png")
    # 画配对图
    sns.pairplot(df[featstr2[15:23]], corner = True)
    fig.savefig("./output/22.png")
    # 尽管相关系数很高，但变量并不完全是线性的，异常值影响了散点图的形状
    # 看其它组的
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[featstr2[23:31]].corr(), center = 0, cmap = "coolwarm", annot = True, cbar = False)
    fig.savefig("./output/23.png")
    # 画配对图
    sns.pairplot(df[featstr2[23:31]], corner = True)
    fig.savefig("./output/24.png")
    # 与前面情况类似
    # 这两组之间是负相关的，画到一起看看
    plt.figure(figsize=(18, 6))
    sns.heatmap(df[featstr2[15:31]].corr(), center = 0, cmap = "coolwarm", annot = True, cbar = False)
    fig.savefig("./output/25.png")
    
    # 下面查看异常值
    # 特征平均值
    fig = px.bar(df[featstr].mean(), title = "Features mean values")
    fig.layout.xaxis.tickangle = 300
    fig.update_traces(showlegend = False)
    fig.layout.xaxis.dtick = 5
    fig.layout.xaxis.title = ""
    fig.layout.yaxis.title = ""
    fig.write_image("./output/26.png")
    # 特征最大值
    fig = px.bar(df[featstr].max(), title = "Features Max values")
    fig.layout.xaxis.tickangle = 300
    fig.update_traces(showlegend = False)
    fig.layout.xaxis.dtick = 5
    fig.layout.xaxis.title = ""
    fig.layout.yaxis.title = ""
    fig.write_image("./output/27.png")
    # 特征最低值
    fig = px.bar(df[featstr].min(), title = "Features min values")
    fig.layout.xaxis.tickangle = 300
    fig.update_traces(showlegend = False)
    fig.layout.xaxis.dtick = 5
    fig.layout.xaxis.title = ""
    fig.layout.yaxis.title = ""
    fig.write_image("./output/28.png")
    plt.close()
    # 上述三个值的分布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    plt.subplots_adjust(hspace=3)
    sns.distplot(df[featstr].max(), ax = ax1)
    sns.distplot(df[featstr].min(), ax = ax2)
    sns.distplot(df[featstr].mean(), ax = ax3)
    fig.suptitle('distribution of mean max and min for features',fontsize=16)
    ax1.set_title('distribution  of features max values',fontsize=14)
    ax1.text(.82,.56,'std = {}'.format(round(df[featstr].max().std(),2)),transform=ax1.transAxes, verticalalignment='top',bbox=props,fontsize=12)
    ax2.set_title('distribution  of features min values',fontsize=14)
    ax2.text(.82,.56,'std = {}'.format(round(df[featstr].min().std(),2)),transform=ax2.transAxes, verticalalignment='top',bbox=props,fontsize=12)
    ax3.set_title('distribution  of features mean values',fontsize=14)
    ax3.text(.82,.56,'std = {}'.format(round(df[featstr].mean().std(),2)),transform=ax3.transAxes, verticalalignment='top',bbox=props,fontsize=12)
    fig.savefig("./output/29.png")
    plt.close()
    # 最大最小值的分布都呈偏态。
    
    # 下面进行更详细的统计描述
    for i in featstr[1:]:
        print('{}\n0.1%:99.9% are between: {}\nmax: {}\nmin: {}\n75% are under: {}'.format(i,np.percentile(df[i],(.1,99.9)), df[i].max(),df[i].min(),np.percentile(df[i],75)), '\n===============================')
        
    print(df[(df.feature_56== df.feature_56.max())|(df.feature_57== df.feature_57.max())|(df.feature_58== df.feature_58.max()) | (df.feature_59== df.feature_59.max())])
    # 可以看出数据有很多极端异常值
    # 一些异常值导致了共线性
    
    # 现在去除超出99.9%特征值的异常值
    n999 = [np.percentile(df[i], 99.9) for i in featstr[1:]]
    n001 = [np.percentile(df[i], 0.1) for i in featstr[1:]]
    
    for i, j in enumerate(featstr[1:]):
        df = df[df[j] < n999[i]]
        # gc.collect()
    # 看看抛弃了多少数据
    print(str(round(((org_len - len(df))/org_len)*100,2))+'%')
    # 再画图看看
    fig = px.bar(df[featstr].max(), title = "Features Max values")
    fig.layout.xaxis.tickangle = 300
    fig.update_traces(showlegend = False)
    fig.layout.xaxis.dtick = 5
    fig.layout.xaxis.title = ""
    fig.layout.yaxis.title = ""
    fig.write_image("./output/30.png")
    plt.close()
    # 再画箱状图
    fig = plt.figure(figsize=(20, 80))
    fig.suptitle('Features Box plot with 0.1% 99.9% whiskers',fontsize=22, y=.89)
    grid = gridspec.GridSpec(29,4,figure=fig,hspace=.5,wspace=.05)
    counter = 0
    for i in range(29):
        for j in range(4):
            subf = fig.add_subplot(grid[i, j])
            sns.boxplot(x = df[featstr[counter]], saturation = 0.5, color = "blue", ax = subf, width = 0.5, whis = (0.1, 99.9))
            subf.set_xlabel("")
            subf.set_title('{}'.format(featstr[counter]),fontsize=16)
            counter += 1
            # gc.collect()
    plt.savefig("./output/31.png")
    plt.close()
    # 正的异常值少了，负的还有，尤其3-40
    # 再处理负的异常值
    for i, j in zip(featstr[1:][2:34], n001[2:34]):
        df = df[df[i] > j]
        # gc.collect()
    # 看去掉了多少
    print(str(round(((org_len - len(df))/org_len)*100,2))+'%')
    # 接着画去除异常值后的概率密度图和柱形图
    fig = plt.figure(figsize=(20,80))
    fig.suptitle('KDE plot of Features',fontsize=24,transform =fig.transFigure, y=.89)
    grid =  gridspec.GridSpec(29,4,figure=fig,hspace=.5,wspace=.01)
    counter = 0
    for i in range(29):
        for j in range(4):
            subf = fig.add_subplot(grid[i, j]);
            sns.distplot(df[df.action==0][featstr[counter]],bins= 100,label='Negative', color='darkorange', kde_kws={'linewidth':4},ax=subf)
        sns.distplot(df[df.action!=0][featstr[counter]],bins= 100,label='Positive', color='blue', kde_kws={'alpha':.9,'linewidth':2},hist_kws={'alpha':.3},ax=subf)
        subf.axvline(np.percentile(df[featstr[counter]],99.5),color= 'darkblue', label='99.5%', linestyle=':',linewidth=2)
        subf.axvline(np.percentile(df[featstr[counter]],.5),color= 'red', label='0.5%', linestyle=':',linewidth=2)
        subf.legend().set_visible(False)
        subf.set_xlabel('')
        subf.set_title('{}'.format(featstr[counter]),fontsize=16)
        kurt=round(df[featstr[counter]].kurt(),2)
        skew=round(df[featstr[counter]].skew(),2)
        subf.text(.6,.92,'Kurt = {:.2f}\nSkew = {:.2f}'.format(kurt ,skew), transform=subf.transAxes, verticalalignment='top',bbox=props, fontsize=10)
        counter += 1
        # gc.collect();
    handles, labels = subf.get_legend_handles_labels()
    fig.legend(handles, labels,ncol=4, bbox_to_anchor=(0.86, 0.893),fontsize=10, title= 'Resp',title_fontsize=14,bbox_transform =fig.transFigure)
    plt.savefig("./output/32.png")
    plt.close()
    """
    通过在特征分布上增加resp值，可以看到：①特征的柱状图现在有办法减少偏差形成更规则的分布。②一些特征比如1,2,85,87,88,91有很多负的偏差值。③一些特征如49,50,51,55,56,57,58,59仍然有正的偏差值。④特征值分布不受resp值的影响。"""
    # 特征与resp的相关性
    # 使用一个Series记录resp与每个特征的关系
    respcorr = pd.Series([df.resp.corr(df[i]) for i in featstr], index = featstr)
    fig = px.bar(respcorr, color = respcorr, color_continuous_scale = ["red", "blue"], title = "Features Correlation with Resp")
    fig.layout.xaxis.tickangle = 300
    fig.layout.xaxis. dtick = 5

    fig.layout.xaxis.title = ''

    fig.layout.yaxis.title = 'pearson correlation'

    fig.update(layout_coloraxis_showscale=False)
    fig.write_image("./output/33.png")
    plt.close()
    # 可以看到feature并不是真的与resp相关
    # 再来看看feature与weight
    # 同样建一个Series，只保留weight大于0的
    wecorr = pd.Series([df[df.weight != 0].weight.corr(df[df.weight != 0][i]) for i in featstr],index=featstr)
    print(wecorr.head(10))
    fig = px.bar(wecorr, color = wecorr, color_continuous_scale = ["red", "blue"], title= 'Features Correlation with Weight (not including zero weights)')
    fig.layout.xaxis.tickangle = 300
    fig.layout.xaxis. dtick = 5

    fig.layout.xaxis.title = ''

    fig.layout.yaxis.title = 'pearson correlation'

    fig.update(layout_coloraxis_showscale=False)
    fig.write_image("./output/34.png")
    plt.close()
    # 现在来看相关性最高的第51号特征，用散点图
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(df[df.weight != 0].weight, df[df.weight != 0].feature_51, color = 'darkblue', alpha=.3)
    plt.xlabel('Weight',fontsize=14)

    plt.ylabel('Featre_51',fontsize=14)

    plt.title('Feature_51 vs Weight\nCorrelation = {}%'.format(round(df[df.weight != 0].weight.corr(df[df.weight != 0].feature_51),4)*100), fontsize=16)
    fig.savefig("./output/35.png")
    plt.close()
    # 看起来51号特征与weigh高度相关
    # 再来看相关度最低的126号特征
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(df[df.weight != 0].weight, df[df.weight != 0].feature_126, color = 'darkblue', alpha=.3)
    plt.xlabel('Weight',fontsize=14)

    plt.ylabel('Featre_126',fontsize=14)

    plt.title('Feature_126 vs Weight\nCorrelation = {}%'.format(round(df[df.weight != 0].weight.corr(df[df.weight != 0].feature_51),4)*100), fontsize=16)
    fig.savefig("./output/36.png")
    plt.close()
    
    # 下面研究特征0
    plt.figure(figsize=(7, 5))
    df.feature_0.value_counts().plot.bar(color='darkblue',alpha=.6,width=.5)
    plt.title('Feature_0',fontsize=18)

    plt.xticks(rotation=0,fontsize=14)
    fig.savefig("./output/37.png")
    plt.close()
    # 看起来是二值分布
    # 再把resp加进来考虑
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="feature_0", hue="action", palette="viridis")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.xlabel('Feature 0',fontsize=12)
    plt.title('Feature 0 and Resp', fontsize=18)
    plt.ylabel('')
    plt.xlim(-1,2)
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h,['Negative','Positive'],ncol=1, fontsize=12, loc=3,title= 'Resp',title_fontsize=14)
    plt.savefig("./output/37.png")
    plt.close()
    # 看起来0号特征为正或为负与resp值无明显关系
    
    # 下面进行降维和群聚
    print("降维")
    scaler = scale()
    scaler.fit(df[featstr[1:]])
    df_pca = pd.DataFrame(scaler.transform(df[featstr[1:]]))
    df_pca.columns = featstr[1:]
    print(df_pca.head())
    # 把数据降维为14个成分
    pca = PCA(n_components=14).fit(df_pca)
    df_pca = pd.DataFrame(pca.transform(df_pca))
    pcs = ["pc" + str(i+1) for i in range(14)]
    # 再把目标列加上去
    df_pca.columns = pcs
    df_pca["action"] = df.action.values
    df_pca["weight"] = df.weight.values
    df_pca["resp"] = df.resp.values
    print(df_pca.head())
    # 画配对图看看
    plt.figure()
    sns.pairplot(data = df_pca, vars = pcs,  hue = "action")
    plt.savefig("./output/38.png")
    plt.close()
    # 降维后也没有明显的关系模式出现
    # 再聚类
    kmeans = k_means(n_clusters = 4, max_iter = 400, random_state = 0, X = df_pca[pcs])
    # 增加聚类列
    df_pca["cluster"] = kmeans[1]
    df_pca["cluster"] = df_pca["cluster"].astype("category")
    print(df_pca.head(8))
    # 画resp与聚类结果的关系
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    sns.countplot(data = df_pca, x = "cluster", hue = "action", ax = ax, palette = "viridis")
    h, l = plt.gca().get_legend_handles_labels()
    plt.legend(h,['Negative','Positive'],ncol=1, fontsize=12, loc=2,title= 'Resp', title_fontsize=14)
    plt.xlim(-1,4)
    plt.xlabel('Clusters',fontsize=16)
    plt.ylabel('')
    plt.title('PCA Clusters and Resp', fontsize=18)
    plt.savefig("./output/39.png")
    plt.close()
    gc.collect()
    
    

# 第二篇文章的EDA
# 参考https://www.kaggle.com/manavtrivedi/eda-and-feature-selection
def EDA2(train_df, test_df, feature_df):
    train = train_df.copy()
    test = test_df.copy()
    print(train.head(10))
    # 看缺失值的情况
    temp = pd.DataFrame(train.isna().sum().sort_values(ascending = False) * 100/train.shape[0], columns = ["missing %"]).head(20)
    print(temp)
    # 新建action列
    train = train[train["weight"] != 0]
    train["action"] = (train["resp"] > 0)*1
    print(train.action.value_counts())
    # 画图看看目标值
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Distributing of weight")
    sns.distplot(train["weight"], color = "blue", kde = True, bins = 100)
    t0 = train[train["action"] == 0]
    t1 = train[train["action"] == 1]
    plt.subplot(1, 2, 2)
    sns.distplot(train['weight'],color='blue',kde=True,bins=100)

    sns.distplot(t0['weight'],color='blue',kde=True,bins=100,label='action = 0')

    sns.distplot(t1['weight'],color='red',kde=True,bins=100,label='action = 1')

    plt.legend()
    plt.savefig("./output/40.png")
    plt.close()
    # 画四个resp对weight的散点图
    fig,ax = plt.subplots(2,2,figsize=(12,10))

    for i,col in enumerate([f'resp_{i}' for i in range(1,5)]):

        plt.subplot(2,2,i+1)

        plt.scatter(train[train.weight!=0].weight,train[train.weight!=0][col])

        plt.ylabel(col)
        plt.xlabel('weight')
    plt.savefig("./output/41.png")
    plt.close()
    
    # 画resp与其余四个resp的图
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    i = 1
    for col in ([f"resp_{i}" for i in range(1, 5)]):
        plt.subplot(2, 2, i)
        plt.plot(train.ts_id.values, train.resp.values, label = "resp", color = "blue")
        plt.plot(train.ts_id.values, train[f"resp_{i}"].values, label = f"resp_{i}", color = "red")
        plt.xlabel("ts_id")
        plt.legend()
        
        i += 1
    plt.savefig("./output/42.png")
    plt.close()
    
    # 画四个resp的散点图
    plt.figure(figsize = (10, 10))
    plt.scatter(train.resp.values, train.resp_1.values, color = "red", label = "resp_1")
    plt.scatter(train.resp.values, train.resp_2.values, color = "blue", label = "resp_2")
    plt.scatter(train.resp.values, train.resp_3.values, color = "orange", label = "resp_3")
    plt.scatter(train.resp.values, train.resp_4.values, color = "green", label = "resp_4")
    plt.xlabel("resp")
    plt.ylabel("other resp variables")
    plt.legend()
    plt.savefig("./output/43.png")
    plt.close()
    
    # 画累积量
    plt.figure(figsize = (8, 6))
    for col in ([f"resp_{i}" for i in range(1, 5)]):
        plt.plot(train[col].cumsum().values, label = col)
    plt.legend()
    plt.title("resp in different time horizons")
    plt.savefig("./output/44.png")
    plt.close()
    
    # 看看feature_0的情况
    sns.countplot(train.feature_0)
    plt.savefig("./output/45.png")
    plt.close()
    
    # 每行的均值的分布，按目标值分类
    features = [col for col in train.columns if "feature" in col]
    t0 = train.loc[train["action"] == 0]
    t1 = train.loc[train["action"] == 1]
    plt.figure(figsize = (16, 6))
    plt.title("Distribution of mean values per row in the train set")
    sns.distplot(t0[features].mean(axis = 1), color = "red", kde = True, bins = 120, label = "target = 0")
    sns.distplot(t1[features].mean(axis = 1), color = "blue", kde = True, bins = 120, label = "target = 1")
    plt.legend()
    plt.savefig("./output/46.png")
    plt.close()
    
    # 每列的均值分布
    plt.figure(figsize = (16, 6))
    plt.title("Distribution of mean values per columns in the train set")
    sns.distplot(t0[features].mean(axis = 0), color = "green", kde = True, bins = 120, label = "target = 0")
    sns.distplot(t1[features].mean(axis = 0), color = "darkblue", kde = True, bins = 120, label = "target = 1")
    plt.legend()
    plt.savefig("./output/47.png")
    plt.close()
    
    # 每行的标准差的分布，按目标值分类
    features = [col for col in train.columns if "feature" in col]
    t0 = train.loc[train["action"] == 0]
    t1 = train.loc[train["action"] == 1]
    plt.figure(figsize = (16, 6))
    plt.title("Distribution of standard deviation values per row in the train set")
    sns.distplot(t0[features].std(axis = 1), color = "red", kde = True, bins = 120, label = "target = 0")
    sns.distplot(t1[features].std(axis = 1), color = "blue", kde = True, bins = 120, label = "target = 1")
    plt.legend()
    plt.savefig("./output/48.png")
    plt.close()
    
    # 每列的标准差分布
    plt.figure(figsize = (16, 6))
    plt.title("Distribution of standard deviation values per columns in the train set")
    sns.distplot(t0[features].std(axis = 0), color = "green", kde = True, bins = 120, label = "target = 0")
    sns.distplot(t1[features].std(axis = 0), color = "darkblue", kde = True, bins = 120, label = "target = 1")
    plt.legend()
    plt.savefig("./output/49.png")
    plt.close()
    
    # 每行的最小值的分布，按目标值分类
    t0 = train.loc[train["action"] == 0]
    t1 = train.loc[train["action"] == 1]
    plt.figure(figsize = (16, 6))
    plt.title("Distribution of min values per row in the train set")
    sns.distplot(t0[features].min(axis = 1), color = "red", kde = True, bins = 120, label = "target = 0")
    sns.distplot(t1[features].min(axis = 1), color = "blue", kde = True, bins = 120, label = "target = 1")
    plt.legend()
    plt.savefig("./output/50.png")
    plt.close()
    
    # 每列的最小值分布
    plt.figure(figsize = (16, 6))
    plt.title("Distribution of min values per columns in the train set")
    sns.distplot(t0[features].min(axis = 0), color = "green", kde = True, bins = 120, label = "target = 0")
    sns.distplot(t1[features].min(axis = 0), color = "darkblue", kde = True, bins = 120, label = "target = 1")
    plt.legend()
    plt.savefig("./output/51.png")
    plt.close()
    
    # 看相关性
    train_corr = train[features].corr().values.flatten()
    train_corr = train_corr[train_corr != 1]
    test_corr = test[features].corr().values.flatten()
    test_corr = test_corr[test_corr != 1]
    
    plt.figure(figsize = (20, 5))
    sns.distplot(train_corr, color = "Red", label = "train")
    sns.distplot(test_corr, color = "Green", label = "test")
    plt.xlabel("Correlation values found in train (except 1)")
    plt.ylabel("Density")
    plt.title("Are there correlations between features?"); 
    plt.legend()
    plt.savefig("./output/52.png")
    plt.close()
    
    # 降维，画解释率
    plt.figure(figsize = (8, 5))
    pca = PCA().fit(train[features].iloc[:, 1:].fillna(train.mean()))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth = 4)
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.xlabel("number of components")
    plt.ylabel("sum of explained variance ratio")
    plt.savefig("./output/53.png")
    plt.close()
    
    # 数据标准化
    rb = RobustScaler()
    data = rb.fit_transform(train[features].iloc[:,1:].fillna(train[features].fillna(train[features].mean())))
    data = PCA(n_components=2).fit_transform(data)
    plt.figure(figsize=(7,7))
    sns.scatterplot(data[:,0],data[:,1], hue=train['action'])
    plt.xlabel('pca comp 1')
    plt.ylabel('pca comp 2')
    plt.savefig("./output/54.png")
    plt.close()
    
    # KNN算法聚类
    X_std = train[[f"feature_{i}" for i in range(1, 130)]].fillna(train.mean()).values
    sse = []
    list_k = list(range(1, 10))
    # 分1-10个类
    for k in list_k:
        km = KMeans(n_clusters = k)
        km.fit(data)
        sse.append(km.inertia_)
        
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.savefig("./output/55.png")
    plt.close()
    
    # 用knn模型预测
    knn = KMeans(n_clusters = 2)
    labels = knn.fit_predict(data)
    sns.scatterplot(data[:, 0], data[:, 1], hue = labels)
    plt.savefig("./output/56.png")
    plt.close()
    
    # 随机森林选出最重要的20个特征
    target = "action"
    cols_drop = list(np.setdiff1d(train.columns, test.columns)) + ["ts_id", "date"]
    
    clf = RandomForestClassifier()
    clf.fit(train.drop(cols_drop, axis = 1).fillna(-999), train["action"])
    top = 20
    top_features = np.argsort(clf.feature_importances_)[::-1][:top]
    feature_names = train.drop(cols_drop, axis = 1).iloc[:, top_features].columns
    
    plt.figure(figsize=(8, 7))
    sns.barplot(clf.feature_importances_[top_features], feature_names, color = "blue")
    plt.savefig("./output/57.png")
    plt.close()
    
    # 画剩下的特征的分布
    """
    top = 8
    top_features = np.argsort(clf.feature_importances_)[::-1][:top]
    feature_names = train.drop(cols_drop, axis = 1).iloc[:, top_features].columns
    
    def plot_features(df1, target = "action", features = []):
        i = 0
        sns.set_style("whitegrid")
        plt.figure()
        fig, ax = plt.subplots(4, 2, figsize=(14, 14))
        
        for feature in features:
            i += 1
            plt.subplot(4, 2, i)
            sns.distplot(df1[df1[target]==1][feature].values,label='1')
            sns.distplot(df1[df1[target]==0][feature].values,label='0')
            plt.xlabel(feature, fontsize = 9)
            plt.legend()
            
        plt.savefig("./output/58.png")
        plt.close()
        
    plot_features(train, features = top_features)
    """
    # 画特征之间的配对图
    sns.pairplot(train[list(feature_names[:10]) + ["action"]], hue = "action")
    plt.savefig("./output/59.png")
    plt.close()
    
    # 计算夏普值?
    import shap
    
    explainer = shap.TreeExplainer(clf)
    X = train.drop(cols_drop, axis = 1).fillna(-999).sample(1000)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type = "bar")
    plt.savefig("./output/60.png")
    plt.close()
    
    shap.dependence_plot("feature_35", shap_values[1], X, display_features = X.sample(1000))
    plt.savefig("./output/61.png")
    plt.close()
    
    
# 第三篇文章的EDA
# 参考https://www.kaggle.com/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance
def EDA3(train, test, feature):
    train_data = train.copy()
    feature_tags = feature.copy()
    
    # 累积收益
    fig, ax = plt.subplots(figsize=(15,5))
    balance= pd.Series(train_data['resp']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_ylabel ("Cumulative resp", fontsize=18);
    balance.plot(lw=3);
    del balance
    plt.savefig("./output/62.png")
    plt.close()
    
    # 四种时间范围，范围越长，越激进，收益也越大。
    fig, ax = plt.subplots(figsize=(15, 5))
    balance= pd.Series(train_data['resp']).cumsum()
    resp_1= pd.Series(train_data['resp_1']).cumsum()
    resp_2= pd.Series(train_data['resp_2']).cumsum()
    resp_3= pd.Series(train_data['resp_3']).cumsum()
    resp_4= pd.Series(train_data['resp_4']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_title ("Cumulative resp and time horizons 1, 2, 3, and 4 (500 days)", fontsize=18)
    balance.plot(lw=3)
    resp_1.plot(lw=3)
    resp_2.plot(lw=3)
    resp_3.plot(lw=3)
    resp_4.plot(lw=3)
    plt.legend(loc="upper left")
    del resp_1
    del resp_2
    del resp_3
    del resp_4
    plt.savefig("./output/63.png")
    plt.close()
    # 可以看出resp与resp_4类似。
    
    # 画所有resp值的柱状图（只显示-0.05-0.05）
    plt.figure(figsize = (12,5))
    ax = sns.distplot(train_data['resp'], 
             bins=3000, 
             kde_kws={"clip":(-0.05,0.05)}, 
             hist_kws={"range":(-0.05,0.05)},
             color='darkcyan', 
             kde=False)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)
    plt.xlabel("Histogram of the resp values", size=14)
    del values
    plt.savefig("./output/64.png")
    plt.close()
    
    # 分布是长尾的
    min_resp = train_data['resp'].min()
    print('The minimum value for resp is: %.5f' % min_resp)
    max_resp = train_data['resp'].max()
    print('The maximum value for resp is:  %.5f' % max_resp)
	# 看这个分布的偏度和峰度
    print("Skew of resp is:      %.2f" %train_data['resp'].skew())
    print("Kurtosis of resp is: %.2f"  %train_data['resp'].kurtosis())
	
	# 下面来看weight值
	# weight为0的值保留在数据里是为了完整性考虑。这个交易对收益无贡献。
	# 看weight为0的比例
    percent_zeros = (100/train_data.shape[0])*((train_data.weight.values == 0).sum())
    print('Percentage of zero weights is: %i' % percent_zeros +"%")
	# 看有没有负值
    min_weight = train_data['weight'].min()
    print('The minimum weight is: %.2f' % min_weight)
	# 最小值为0，没有负值。
	# 再来看最大值
    max_weight = train_data['weight'].max()
    print('The maximum weight was: %.2f' % max_weight)
    # 看在哪天 第446天
    print(train_data[train_data['weight']==train_data['weight'].max()])
	
	# 看非0的weight值的分布
    plt.figure(figsize = (12,5))
    ax = sns.distplot(train_data['weight'], 
             bins=1400, 
             kde_kws={"clip":(0.001,1.4)}, 
             hist_kws={"range":(0.001,1.4)},
             color='darkcyan', 
             kde=False)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)
    plt.xlabel("Histogram of non-zero weights", size=14)
    plt.savefig("./output/65.png")
    plt.close()
    del values
    # 有两个峰值，会是两个分布叠加吗？
	
    # 画weight值的对数分布
    train_data_nonZero = train_data.query('weight > 0').reset_index(drop = True)
    plt.figure(figsize = (10,4))
    ax = sns.distplot(np.log(train_data_nonZero['weight']), 
             bins=1000, 
             kde_kws={"clip":(-4,5)}, 
             hist_kws={"range":(-4,5)},
             color='darkcyan', 
             kde=False)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)
    plt.xlabel("Histogram of the logarithm of the non-zero weights", size=14)
    plt.savefig("./output/66.png")
    plt.close()
    
    # 用高斯分布等来拟合
    from scipy.optimize import curve_fit
    # the values
    x = list(range(len(values)))
    x = [(i/110)-4 for i in x]
    y = values

    # define a Gaussian function
    def Gaussian(x,mu,sigma,A):
    	return A*np.exp(-0.5 * ((x-mu)/sigma)**2)

    def bimodal(x,mu_1,sigma_1,A_1,mu_2,sigma_2,A_2):
    	return Gaussian(x,mu_1,sigma_1,A_1) + Gaussian(x,mu_2,sigma_2,A_2)

    # seed guess
    initial_guess=(1, 1 , 1,    1, 1, 1)

    # the fit
    parameters,covariance=curve_fit(bimodal,x,y,initial_guess)
    sigma=np.sqrt(np.diag(covariance))
    # the plot
    plt.figure(figsize = (10,4))
    ax = sns.distplot(np.log(train_data_nonZero['weight']),              bins=1000, kde_kws={"clip":(-4,5)},hist_kws={"range":(-4,5)}, color='darkcyan', kde=False)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)
    plt.xlabel("Histogram of the logarithm of the non-zero weights", size=14)
    # plot gaussian #1
    plt.plot(x,Gaussian(x,parameters[0],parameters[1],parameters[2]),':',color='black',lw=2,label='Gaussian #1', alpha=0.8)
    # plot gaussian #2
    plt.plot(x,Gaussian(x,parameters[3],parameters[4],parameters[5]),'--',color='black',lw=2,label='Gaussian #2', alpha=0.8)
    # plot the two gaussians together
    plt.plot(x,bimodal(x,*parameters),color='black',lw=2, alpha=0.7)
    plt.legend(loc="upper left");
    plt.savefig("./output/67.png")
    plt.close()
    del values
    # 左边的那个峰值似乎是其它分布的。
	
    # 累积回报
    # weight和resp的和构成收益。
    train_data['weight_resp'] = train_data['weight']*train_data['resp']
    train_data['weight_resp_1'] = train_data['weight']*train_data['resp_1']
    train_data['weight_resp_2'] = train_data['weight']*train_data['resp_2']
    train_data['weight_resp_3'] = train_data['weight']*train_data['resp_3']
    train_data['weight_resp_4'] = train_data['weight']*train_data['resp_4']
    fig, ax = plt.subplots(figsize=(15, 5))
    resp = pd.Series(1+(train_data.groupby('date')['weight_resp'].mean())).cumprod()
    resp_1  = pd.Series(1+(train_data.groupby('date')['weight_resp_1'].mean())).cumprod()
    resp_2  = pd.Series(1+(train_data.groupby('date')['weight_resp_2'].mean())).cumprod()
    resp_3  = pd.Series(1+(train_data.groupby('date')['weight_resp_3'].mean())).cumprod()
    resp_4  = pd.Series(1+(train_data.groupby('date')['weight_resp_4'].mean())).cumprod()
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_title ("Cumulative daily return for resp and time horizons 1, 2, 3, and 4 (500 days)", fontsize=18)
    resp.plot(lw=3, label='resp x weight')
    resp_1.plot(lw=3, label='resp_1 x weight')
    resp_2.plot(lw=3, label='resp_2 x weight')
    resp_3.plot(lw=3, label='resp_3 x weight')
    resp_4.plot(lw=3, label='resp_4 x weight')
    # day 85 marker
    ax.axvline(x=85, linestyle='--', alpha=0.3, c='red', lw=1)
    ax.axvspan(0, 85 , color=sns.xkcd_rgb['grey'], alpha=0.1)
    plt.legend(loc="lower left");
    plt.savefig("./output/68.png")
    plt.close()
    # 对于较短期的策略，resp_1 - resp_3，收益较低。
    
    # 现在画weight和resp之积（排除weight=0的）
    train_data_no_0 = train_data.query('weight > 0').reset_index(drop = True)
    train_data_no_0['wAbsResp'] = train_data_no_0['weight'] * (train_data_no_0['resp'])
    #plot
    plt.figure(figsize = (12,5))
    ax = sns.distplot(train_data_no_0['wAbsResp'], 
             bins=1500, 
             kde_kws={"clip":(-0.02,0.02)}, 
             hist_kws={"range":(-0.02,0.02)},
             color='darkcyan', 
             kde=False)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
    	rec.set_color(col)
    plt.xlabel("Histogram of the weights * resp", size=14)
    plt.savefig("./output/69.png")
    plt.close()
    
    # 时间
    # 画每天的ts_id数量
    trades_per_day = train_data.groupby(['date'])['ts_id'].count()
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(trades_per_day)
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_title ("Total number of ts_id for each day", fontsize=18)
    # day 85 marker
    ax.axvline(x=85, linestyle='--', alpha=0.3, c='red', lw=1)
    ax.axvspan(0, 85 , color=sns.xkcd_rgb['grey'], alpha=0.1)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=500)
    plt.savefig("./output/70.png")
    plt.close()
    # 在85天那里画了线，是想看85天之后是否改变了交易策略。
    
    # 假设每个交易日6.5小时，看平均每天的交易次数
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(23400/trades_per_day)
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_ylabel ("Av. time between trades (s)", fontsize=18)
    ax.set_title ("Average time between trades for each day", fontsize=18)
    ax.axvline(x=85, linestyle='--', alpha=0.3, c='red', lw=1)
    ax.axvspan(0, 85 , color=sns.xkcd_rgb['grey'], alpha=0.1)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=500)
    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=12)
    plt.savefig("./output/71.png")
    plt.close()
    
    # 每个交易日的交易次数的分布
    plt.figure(figsize = (12,4))
    # the minimum has been set to 1000 so as not to draw the partial days like day 2 and day 294
    # the maximum number of trades per day is 18884
    # I have used 125 bins for the 500 days
    ax = sns.distplot(trades_per_day, 
             bins=125, 
             kde_kws={"clip":(1000,20000)}, 
             hist_kws={"range":(1000,20000)},
             color='darkcyan', 
             kde=True)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)
    plt.xlabel("Number of trades per day", size=14)
    plt.savefig("./output/72.png")
    plt.close()
    # 注意到feature_64看起来像某类每日时钟
    
    # 来探索特征
    # feature_0很特别，全是1和-1
    print(train_data['feature_0'].value_counts())
    # feature_0是在features.csv中唯一有非True标签的特征
    fig, ax = plt.subplots(figsize=(15, 4))
    feature_0 = pd.Series(train_data['feature_0']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_ylabel ("feature_0 (cumulative)", fontsize=18)
    feature_0.plot(lw=3)
    plt.savefig("./output/73.png")
    plt.close()
    
    # 单独画feature_0 = 1和-1时的累计收益
    feature_0_is_plus_one  = train_data.query('feature_0 ==  1').reset_index(drop = True)
    feature_0_is_minus_one = train_data.query('feature_0 == -1').reset_index(drop = True)
    # the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    ax1.plot((pd.Series(feature_0_is_plus_one['resp']).cumsum()), lw=3, label='resp')
    ax1.plot((pd.Series(feature_0_is_plus_one['resp']*feature_0_is_plus_one['weight']).cumsum()), lw=3, label='return')
    ax2.plot((pd.Series(feature_0_is_minus_one['resp']).cumsum()), lw=3, label='resp')
    ax2.plot((pd.Series(feature_0_is_minus_one['resp']*feature_0_is_minus_one['weight']).cumsum()), lw=3, label='return')
    ax1.set_title ("feature 0 = 1", fontsize=18)
    ax2.set_title ("feature 0 = -1", fontsize=18)
    ax1.legend(loc="lower left")
    ax2.legend(loc="upper left");
    plt.savefig("./output/74.png")
    plt.close()
    del feature_0_is_plus_one
    del feature_0_is_minus_one
    # 两种不同的收益情况。
	
    # 看起来有四种不同的特征，各用一个例子画出来。
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,10))
    ax1.plot((pd.Series(train_data['feature_1']).cumsum()), lw=3, color='red')
    ax1.set_title ("Linear", fontsize=22)
    ax1.axvline(x=514052, linestyle='--', alpha=0.3, c='green', lw=2)
    ax1.axvspan(0, 514052 , color=sns.xkcd_rgb['grey'], alpha=0.1)
    ax1.set_xlim(xmin=0)
    ax1.set_ylabel ("feature_1", fontsize=18)

    ax2.plot((pd.Series(train_data['feature_3']).cumsum()), lw=3, color='green')
    ax2.set_title ("Noisy", fontsize=22)
    ax2.axvline(x=514052, linestyle='--', alpha=0.3, c='red', lw=2)
    ax2.axvspan(0, 514052 , color=sns.xkcd_rgb['grey'], alpha=0.1)
    ax2.set_xlim(xmin=0)
    ax2.set_ylabel ("feature_3", fontsize=18)

    ax3.plot((pd.Series(train_data['feature_55']).cumsum()), lw=3, color='darkorange')
    ax3.set_title ("Hybryd (Tag 21)", fontsize=22);
    ax3.set_xlabel ("Trade", fontsize=18)
    ax3.axvline(x=514052, linestyle='--', alpha=0.3, c='green', lw=2)
    ax3.axvspan(0, 514052 , color=sns.xkcd_rgb['grey'], alpha=0.1)
    ax3.set_xlim(xmin=0)
    ax3.set_ylabel ("feature_55", fontsize=18)

    ax4.plot((pd.Series(train_data['feature_73']).cumsum()), lw=3, color='blue')
    ax4.set_title ("Negative", fontsize=22)
    ax4.set_xlabel ("Trade", fontsize=18)
    ax4.set_ylabel ("feature_73", fontsize=18)
	
    plt.savefig("./output/75.png")
    plt.close()
    
    # 标签14很有趣，其特征在整个交易日内只有离散值。（是股票价格吗？)
    # 画散点图看看
    day_0 = train_data.loc[train_data['date'] == 0]
    day_1 = train_data.loc[train_data['date'] == 1]
    day_3 = train_data.loc[train_data['date'] == 3]
    three_days = pd.concat([day_0, day_1, day_3])
    three_days.plot.scatter(x='ts_id', y='feature_41', s=0.5, figsize=(15,3))
    three_days.plot.scatter(x='ts_id', y='feature_42', s=0.5, figsize=(15,3))
    three_days.plot.scatter(x='ts_id', y='feature_43', s=0.5, figsize=(15,3))
    #del day_0
    del day_1
    del day_3
    plt.savefig("./output/76.png")
    plt.close()
    
    # 延迟画图（不知道啥意思）
    from pandas.plotting import lag_plot
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    lag_plot(day_0['feature_41'], lag=1, s=0.5, ax=ax[0])
    lag_plot(day_0['feature_42'], lag=1, s=0.5, ax=ax[1])
    lag_plot(day_0['feature_43'], lag=1, s=0.5, ax=ax[2])
    ax[0].title.set_text('feature_41')
    ax[0].set_xlabel("ts_id (n)")
    ax[0].set_ylabel("ts_id (n+1)")
    ax[1].title.set_text('feature_42')
    ax[1].set_xlabel("ts_id (n)")
    ax[1].set_ylabel("ts_id (n+1)")
    ax[2].title.set_text('feature_43')
    ax[2].set_xlabel("ts_id (n)")
    ax[2].set_ylabel("ts_id (n+1)")

    ax[0].plot(0, 0, 'r.', markersize=15.0)
    ax[1].plot(0, 0, 'r.', markersize=15.0)
    ax[2].plot(0, 0, 'r.', markersize=15.0)
	
    plt.savefig("./output/77.png")
    plt.close()
    
    # Tag22 特征60-68
    fig, ax = plt.subplots(figsize=(15, 5))
    feature_60= pd.Series(train_data['feature_60']).cumsum()
    feature_61= pd.Series(train_data['feature_61']).cumsum()
    feature_62= pd.Series(train_data['feature_62']).cumsum()
    feature_63= pd.Series(train_data['feature_63']).cumsum()
    feature_64= pd.Series(train_data['feature_64']).cumsum()
    feature_65= pd.Series(train_data['feature_65']).cumsum()
    feature_66= pd.Series(train_data['feature_66']).cumsum()
    feature_67= pd.Series(train_data['feature_67']).cumsum()
    feature_68= pd.Series(train_data['feature_68']).cumsum()
    #feature_69= pd.Series(train_data['feature_69']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_title ("Cumulative plot for feature_60 ... feature_68 (Tag 22).", fontsize=18)
    feature_60.plot(lw=3)
    feature_61.plot(lw=3)
    feature_62.plot(lw=3)
    feature_63.plot(lw=3)
    feature_64.plot(lw=3)
    feature_65.plot(lw=3)
    feature_66.plot(lw=3)
    feature_67.plot(lw=3)
    feature_68.plot(lw=3)
    #feature_69.plot(lw=3)
    plt.legend(loc="upper left")
    del feature_60, feature_61, feature_62, feature_63, feature_64, feature_65, feature_66 ,feature_67, feature_68
    plt.savefig("./output/78.png")
    plt.close()
    # 可以看到这些特征很相似。
    # 现在画直方图看看分布
    sns.set_palette("bright")
    fig, axes = plt.subplots(2,2,figsize=(8,8))
    sns.distplot(train_data[['feature_60']], hist=True, bins=200,  ax=axes[0,0])
    sns.distplot(train_data[['feature_61']], hist=True, bins=200,  ax=axes[0,0])
    axes[0,0].set_title ("features 60 and 61", fontsize=18)
    axes[0,0].legend(labels=['60', '61'])
    sns.distplot(train_data[['feature_62']], hist=True,  bins=200, ax=axes[0,1])
    sns.distplot(train_data[['feature_63']], hist=True,  bins=200, ax=axes[0,1])
    axes[0,1].set_title ("features 62 and 63", fontsize=18)
    axes[0,1].legend(labels=['62', '63'])
    sns.distplot(train_data[['feature_65']], hist=True,  bins=200, ax=axes[1,0])
    sns.distplot(train_data[['feature_66']], hist=True,  bins=200, ax=axes[1,0])
    axes[1,0].set_title ("features 65 and 66", fontsize=18)
    axes[1,0].legend(labels=['65', '66'])
    sns.distplot(train_data[['feature_67']], hist=True,  bins=200, ax=axes[1,1])
    sns.distplot(train_data[['feature_68']], hist=True,  bins=200, ax=axes[1,1])
    axes[1,1].set_title ("features 67 and 68", fontsize=18)
    axes[1,1].legend(labels=['67', '68'])
    plt.savefig("./output/79.png")
    plt.close()
    
    # 它们之间是feature_64
    plt.figure(figsize = (12,5))
    ax = sns.distplot(train_data['feature_64'], 
             bins=1200, 
             kde_kws={"clip":(-6,6)}, 
             hist_kws={"range":(-6,6)},
             color='darkcyan', 
             kde=False)
    values = np.array([rec.get_height() for rec in ax.patches])
    norm = plt.Normalize(values.min(), values.max())
    colors = plt.cm.jet(norm(values))
    for rec, col in zip(ax.patches, colors):
    	rec.set_color(col)
    plt.xlabel("Histogram of feature_64", size=14)
    plt.savefig("./output/80.png")
    plt.close()
    del values
    # 在0.7-1.38之间有间隙，像是(ln(2)=0.693, ln(4)=1.386)
	
    # Tag22还有很明显的每天的间隔，如feature64
    day_0 = train_data.loc[train_data['date'] == 0]
	
    day_1 = train_data.loc[train_data['date'] == 1]
    day_3 = train_data.loc[train_data['date'] == 3]
    three_days = pd.concat([day_0, day_1, day_3])

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    ax[0].scatter(three_days.ts_id, three_days.feature_64, s=0.5, color='b')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('value')
    ax[0].set_title('feature_64 (days 0, 1 and 3)')
    ax[1].scatter(three_days.ts_id, pd.Series(three_days['feature_64']).cumsum(), s=0.5, color='r')
    ax[1].set_xlabel('ts_id')
    ax[1].set_ylabel('cumulative sum')
    ax[1].set_title('')
    plt.savefig("./output/81.png")
    plt.close()
    # feature_64的全局最小值为-6.4，全局最大值为8，猜测其单位是30分钟
    # 用arcsin试试
    x = np.arange(-1, 1, 0.01)
    y = 2*np.arcsin(x)+1
    fig, ax = plt.subplots(1, 1, figsize=(7,4))
    ax.plot(x, y, lw=3)
    ax.set(xticklabels = [])
    ax.set(yticklabels = ['9:00','10:00','11:00','12:00','13:00','14:00','15:00' ,'16:00'])
    ax.set_title("2$\it{arcsin}$(x) +1", fontsize=18)
    plt.savefig("./output/82.png")
    plt.close()
	
    # 再画feature_65看看
    three_days.plot.scatter(x='ts_id', y='feature_65', s=0.5, figsize=(15,4))
	
    # 再来看“Noisy”特征
    fig, ax = plt.subplots(figsize=(15, 5))
    feature_3= pd.Series(train_data['feature_3']).cumsum()
    feature_4= pd.Series(train_data['feature_4']).cumsum()
    feature_5= pd.Series(train_data['feature_5']).cumsum()
    feature_6= pd.Series(train_data['feature_6']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_title ("Cumulative plot for features 3, 4, 5 and 6", fontsize=18)
    ax.axvline(x=514052, linestyle='--', alpha=0.3, c='black', lw=1)
    ax.axvspan(0,  514052, color=sns.xkcd_rgb['grey'], alpha=0.1)
    #ax.set_xlim(xmin=0)
    feature_3.plot(lw=3)
    feature_4.plot(lw=3)
    feature_5.plot(lw=3)
    feature_6.plot(lw=3)
    plt.legend(loc="upper left")
    plt.savefig("./output/83.png")
    plt.close()
    
    # Tag19 feature_51
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.scatter(train_data_nonZero.weight, train_data_nonZero.feature_51, s=0.1, color='b')
    ax.set_xlabel('weight')
    ax.set_ylabel('feature_51')
    plt.savefig("./output/84.png")
    plt.close()
	
    # Tag19 feature_52
    fig, ax = plt.subplots(figsize=(15, 3))
    feature_0 = pd.Series(train_data['feature_52']).cumsum()
    ax.set_xlabel ("ts_id", fontsize=18)
    ax.set_ylabel ("feature_52 (cumulative)", fontsize=12)
    feature_0.plot(lw=3)
    plt.savefig("./output/84.png")
    plt.close()
	
    # 延迟画图
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    lag_plot(day_0['feature_52'], s=0.5, ax=ax)
    ax.title.set_text('feature_52')
    ax.set_xlabel("ts_id (n)")
    ax.set_ylabel("ts_id (n+1)")
    ax.plot(0, 0, 'r.', markersize=15.0)
    plt.savefig("./output/85.png")
    plt.close()
	
    # "Negative"特征
    fig, ax = plt.subplots(figsize=(15, 5))
    feature_55= pd.Series(train_data['feature_55']).cumsum()
    feature_56= pd.Series(train_data['feature_56']).cumsum()
    feature_57= pd.Series(train_data['feature_57']).cumsum()
    feature_58= pd.Series(train_data['feature_58']).cumsum()
    feature_59= pd.Series(train_data['feature_59']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_title ("Cumulative plot for the 'Tag 21' features (55-59)", fontsize=18)
    ax.axvline(x=514052, linestyle='--', alpha=0.3, c='black', lw=1)
    ax.axvspan(0,  514052, color=sns.xkcd_rgb['grey'], alpha=0.1)
    feature_55.plot(lw=3)
    feature_56.plot(lw=3)
    feature_57.plot(lw=3)
    feature_58.plot(lw=3)
    feature_59.plot(lw=3)
    plt.legend(loc="upper left")
    plt.savefig("./output/86.png")	
    plt.close()
	
    # 再来看看features.csv
    # 先画图
    #plt.figure(figsize=(32,14))
#    sns.heatmap(feature_tags.T)
#    plt.savefig("./output/87.png")
#    plt.close()
	
    # 看看标签总数
    tag_sum = pd.DataFrame(feature_tags.T.sum(axis=0),columns=['Number of tags'])
    print(tag_sum.T)
    # 可以看到所有特征都有至少一个标签，一些有4个。feature_0没有标签
	
    # 下面看Action
    # 先增加该列
    train_data['action'] = ((train_data['resp'])>0)*1
    # 看看统计情况
    print(train_data["action"].value_counts())
    # 行动比不行动稍微好一点:0.4%，看看每天的情况
    daily_action_sum = train_data['action'].groupby(train_data['date']).sum()
    daily_action_count = train_data['action'].groupby(train_data['date']).count()
    daily_ratio = daily_action_sum/daily_action_count
    # now plot
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(daily_ratio)
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_ylabel ("ratio", fontsize=18)
    ax.set_title ("Daily ratio of action to inaction", fontsize=18)
    plt.axhline(0.5, linestyle='--', alpha=0.85, c='r');
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=500)
    plt.savefig("./output/88.png")
    plt.close()
	
    # 分布非常均匀，没有特别的模式。
    daily_ratio_mean = daily_ratio.mean()
    print('The mean daily ratio is %.3f' % daily_ratio_mean)
    daily_ratio_max = daily_ratio.max()
    print('The maximum daily ratio is %.3f' % daily_ratio_max)
	
    # 现在来看看第0天
    day_0 = train_data.loc[train_data["date"] == 0]
    fig, ax = plt.subplots(figsize=(15, 5))
    balance= pd.Series(day_0['resp']).cumsum()
    resp_1= pd.Series(day_0['resp_1']).cumsum()
    resp_2= pd.Series(day_0['resp_2']).cumsum()
    resp_3= pd.Series(day_0['resp_3']).cumsum()
    resp_4= pd.Series(day_0['resp_4']).cumsum()
    ax.set_xlabel ("Trade", fontsize=18)
    ax.set_title ("Cumulative values for resp and time horizons 1, 2, 3, and 4 for day 0", fontsize=18)
    balance.plot(lw=3)
    resp_1.plot(lw=3)
    resp_2.plot(lw=3)
    resp_3.plot(lw=3)
    resp_4.plot(lw=3)
    plt.legend(loc="upper left")
    plt.savefig("./output/89.png")
    plt.close()
	
    # 第0天的train.csv数据的统计学描述
    print(day_0.describe())
	
    # 看缺失值
    import missingno as msno
    msno.matrix(day_0, color=(0.35, 0.35, 0.75))
    plt.savefig("./output/90.png")
    plt.close()
    
    feats_7_11 = day_0.iloc[:, [14,18]]
    msno.matrix(feats_7_11, color=(0.35, 0.35, 0.75), width_ratios=(1, 3))
    plt.savefig("./output/91.png")
    plt.close()
    # 可以看到缺失值并不是随机分布的，中间有大段缺失值。
	
    # 下面看所有列的缺失值信息
    gone = train_data.isnull().sum()
    px.bar(gone, color=gone.values, title="Total number of missing values for each column").write_image("./output/92.png")
    # 79.6%的缺失值在Tag4组，代表resp_1特征
    # 15.2%的缺失值在Tag3组，代表resp_2特征
    # 上述两者包括了大于95%的缺失数据。
    # 很多特征有相同数量的缺失值。
    # 画图看看
    missing_features = train_data.iloc[:,7:137].isnull().sum(axis=1).groupby(train_data['date']).sum().to_frame()
    # now make a plot
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(missing_features)
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_title ("Total number of missing values in all features for each day", fontsize=18)
    ax.axvline(x=85, linestyle='--', alpha=0.3, c='red', lw=2)
    ax.axvspan(0,  85, color=sns.xkcd_rgb['grey'], alpha=0.1)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=500)
    plt.savefig("./output/93.png")
    plt.close()
    
    # 画每个交易的平均缺失特征数量
    count_weights = train_data[['date', 'weight']].groupby('date').agg(['count'])
    result = pd.merge(count_weights, missing_features, on = "date", how = "inner")
    result.columns = ['weights','missing']
    result['ratio'] = result['missing']/result['weights']
    missing_per_trade = result['ratio'].mean()

    # now make a plot
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(result['ratio'])
    plt.axhline(missing_per_trade, linestyle='--', alpha=0.85, c='r');
    ax.set_xlabel ("Day", fontsize=18)
    ax.set_title ("Average number of missing feature values per trade, for each day", fontsize=18)
    plt.savefig("./output/94.png")
    plt.close()
    # 平均每个交易约有3个缺失特征。
    
    # 采用持续重要性计算(permutation importance calculation)
    X_train = day_0.loc[:, day_0.columns.str.contains('feature')]
    X_train = X_train.fillna(X_train.mean())
    # our target is the action
    y_train = day_0['resp']

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(max_features='auto')
    regressor.fit(X_train, y_train)



if __name__ == "__main__":
    # newpath = "/home/code"
    # os.chdir(newpath)

    
    # 真正开始干活
    data_explore()
