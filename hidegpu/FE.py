# coding:utf-8
# kaggle竞赛Jane Street Market Prediction
# 特征工程代码


import pandas as pd
import matplotlib.pyplot as plt


"""
# 特征工程
@change_dir
def featureEngineer(data):
    tages = pd.DataFrame()
    tagename = feature.columns
    for i in range(29):
        # tagename = "tag_" + str(i)
        # tages[tagename[i+1]] = feature[(feature[tagename[i+1]] == True)].iloc[:, i+1]
        #print(tages[i])
        temp = feature["feature"][feature[tagename[i+1]] == True]
        temp.name = tagename[i+1]
        print(temp)
    #print(tages)
    # 填充空值
    print(data.isnull().sum())
    for col in data.columns:
        mean_val = data[col].mean()
        data[col].fillna(mean_val, inplace=True)
    print(data.isnull().sum())
    # 处理feature_0
    feature_0 = data["feature_0"].cumsum()
    plt.plot(feature_0)
    plt.savefig("./output/cumf_0.png")
    plt.close()
    data["feature_0"] = feature_0
    # print(feature_0)
    return data
"""
# 特征工程
def featureEngineer(data):
    # data = data[data['weight'] != 0]
    data = data.fillna(0.0)
    weight = data['weight'].values
    resp = data['resp'].values
    data['action'] = ((weight * resp) > 0).astype('int')
    return data
    

    
    
if __name__ == "__main__":
    train, feature = loadData()
    # feature = feature[feature == True]
    print(feature)
    train = featureEngineer(train)
    
