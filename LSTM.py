# coding:utf-8
# kaggle Jane Street Market Prediction代码
"""
用pytorch实现LSTM模型
参考:https://zhuanlan.zhihu.com/p/104475016
"""


import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from run import *


class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size = 1, output_size = 1, num_layers = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
        
    def forward(self, _x):
        x, _ = self.lstm(_x)
        s, b, h = x.shape # seq_len, batch, hidden_size
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x
        
        
@change_dir
def LSTM():
    # 建立数据
    data_len = 200
    t = np.linspace(0, 12*np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = sin_t
    dataset[:, 1] = cos_t
    dataset = dataset.astype("float32")
    
    # 划分数据
    train_data_ratio = 0.5
    train_data_len = int(data_len*train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]
    
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]
    
    # 训练
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM) # 分5批
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM) # 分5批
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size = OUTPUT_FEATURES_NUM, num_layers = 1)
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    
    loss_fn = nn.MSELoss()
    lr = 1e-2
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr)
    
    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_fn(output, train_y_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            
    # 用模型预测
    # 训练集上
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
    
    # 切换为测试状态
    lstm_model = lstm_model.eval()
    # 用测试集预测
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM) 
    test_x_tensor = torch.from_numpy(test_x_tensor)
    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()
    
    # 画图
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line') # separation line

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size = 15, alpha = 1.0)
    plt.text(20, 2, "test", size = 15, alpha = 1.0)
    
    plt.savefig("./output/LSTM.png")


if __name__ == "__main__":
    LSTM()