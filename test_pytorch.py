# coding:utf-8
# kaggle Jane Street Market Prediction代码
# 学习pytorch
# 参考https://pytorch.apachecn.org/docs/1.0/pytorch_with_examples.html


from run import *
import matplotlib.pyplot as plt

# 用numpy实现
import numpy as np


# 前向传播
def fp_np(x, w1, w2):
    # 向前传播，计算预测值
     h = x.dot(w1)
     h_relu = np.maximum(h, 0)
     y_pred = h_relu.dot(w2)
     return y_pred, h_relu, h
     
     
# 反向传播
def bp_np(x, y, y_pred, h_relu, h, w1, w2):
    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    return w1, w2
    
    
def nn_numpy():
    print("numy版神经网络")
    # N是批大小；D_in是输入维度
    # H是隐藏层维度；D_out是输出维度
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    # 产生随机输入和输出数据
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    print(len(x))
    print(len(y))
    
    # 随机初始化权重
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)
    learning_rate = 1e-6
    
    for t in range(500):
        # 向前传播，计算预测值
        y_pred, h_relu, h = fp_np(x, w1, w2)
        
        # 计算并显示loss(损失)
        loss = np.square(y_pred - y).sum()
        # print(t, loss)
        
        # 反向传播，计算w1,w2对loss的梯度
        grad_w1, grad_w2 = bp_np(x, y, y_pred, h_relu, h, w1, w2)
        
        # 更新权重
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        
    x_test = np.random.randn(N, D_in)
    print(fp_np(x_test, w1, w2))
    
    
# 用pytorch实现
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
from torchviz import make_dot


def nn_pytorch():
    print("pytorch版神经网络")
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    
    # 产生随机权重tensor
    w1 = torch.randn(D_in, H, device=device, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, requires_grad=True)
    
    learning_rate = 1e-6
    for t in range(500):
        # 前向传播，自动计算梯度
        y_pred = x.mm(w1).clamp(min = 0).mm(w2)
        # 计算并输出loss
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())
        # 反向传播
        loss.backward()
        
        # 更新权重，不自动计算梯度
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad
            
            # 梯度置零
            w1.grad.zero_()
            w2.grad.zero_()
    x_test = torch.randn(N, D_in)
    print(x_test.mm(w1).clamp(min = 0).mm(w2))
    
    
# 用pytorch.nn实现
def nn_torch_nn():
    print("pytorch_nn版神经网络")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in, device=device)
    y = torch.randn(N, D_out, device=device)
    
    model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
    ).to(device)
    
    loss_fn = torch.nn.MSELoss(reduction = "sum")
    
    learning_rate = 1e-4
    for t in range(500):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        print(t, loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad
    
    x_test = torch.randn(N, D_in)
    print(model(x_test))
    
    
# Pytorch实现二分类器
def pytorch_class():
    class ClassifyModel(nn.Module):
        def __init__(self, input_dim, hide_dim, output_dim):
            super(ClassifyModel, self).__init__()
            self.linear1 = nn.Linear(input_dim, hide_dim)
            self.linear2 = nn.Linear(hide_dim, output_dim)
            
        def forward(self, x):
            hidden = self.linear1(x)
            activate = torch.relu(hidden)
            output = self.linear2(activate)
            return output
            
    # 准备数据
    x = torch.unsqueeze(torch.linspace(-10, 10, 50), 1)
    y = torch.cat((torch.ones(25), torch.zeros(25))).type(torch.LongTensor)
    print(x)
    print(y)
    dataset = Data.TensorDataset(x, y)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=5, shuffle=True)
    model = ClassifyModel(1, 10, 2)
    model2 = torch.nn.Sequential(
             nn.Linear(1, 10),
             nn.ReLU(),
             nn.Linear(10, 2),
     )
     
    optim = torch.optim.Adam(model2.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
     
    for e in range(1000):
        epoch_loss = 0
        epoch_acc = 0
        for i, (x, y) in enumerate(dataloader):
            optim.zero_grad()
            out = model2(x)
            loss = loss_fn(out, y)
             
            loss.backward()
            optim.step()
             
            epoch_loss += loss.data
            epoch_acc += get_acc(out, y)
             
            if e % 200 == 0:
                print('epoch: %d, loss: %f, acc: %f' % (e, epoch_loss / 50, epoch_acc / 50))
                
    x_test = torch.unsqueeze(torch.linspace(-2, 2, 10), 1)
    print(x_test)
    y_pred = (model2(x_test))
    print(y_pred)
            
            
def get_acc(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num
    return acc
    
    
# 新的尝试
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
@change_dir
def new_try():
    # 1.一个简单的回归问题
    # 生成数据
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2*x + 0.1*np.random.randn(100, 1)
    # 打乱顺序
    idx = np.arange(100)
    np.random.shuffle(idx)
    # 使用前80个数据做训练集
    train_idx = idx[:80]
    # 剩下的做验证集
    val_idx = idx[80:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[val_idx], y[val_idx]
    plt.figure()
    plt.scatter(x_train, y_train)
    plt.savefig("./output/train.png")
    plt.close()
    plt.figure()
    plt.scatter(x_test, y_test)
    plt.savefig("./output/test.png")
    plt.close()
    
    # 2.梯度下降
    # 第一步，计算损失值loss
    # 对于回归问题，用平均方差
    # Mean Square Error (MSE)
    # 第二步，计算梯度
    # 即当我们轻微变动两个参数a,b时MSE如何变化
    # 第三步，更新参数
    # 第四步，用新的参数重新进行上述步骤
    # 这个过程就是训练模型的过程
    
    # 3.使用numpy进行线性回归
    # 初始化步骤有两步
    # ①随机初始化参数和权重
    np.random.seed(42)
    a = np.random.randn(1)
    b = np.random.randn(1)
    print(a, b)
    # ②初始化超参数
    lr = 1e-1
    n_epochs = 1000
    
    # 训练过程
    for epoch in range(n_epochs):
        # 计算模型预测值:前向传播
        yhat = a + b*x_train
        # 计算损失值
        error = (y_train - yhat)
        loss = (error**2).mean()
        # 计算每个参数的梯度值
        a_grad = -2*error.mean()
        b_grad = -2*(x_train*error).mean()
        # 使用梯度和学习率更新参数
        a -= lr*a_grad
        b -= lr*b_grad
        
    print(a, b)
    
    # 检查一下对不对
    from sklearn.linear_model import LinearRegression
    linr = LinearRegression()
    linr.fit(x_train, y_train)
    print(linr.intercept_, linr.coef_[0])
    
    # 4.使用pytorch
    # 张量tensor，有三个或更多的维度
    # 加载数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    print(type(x_train), type(x_train_tensor), x_train_tensor.type())
    # 创建参数
    # 第一种方法
    a = torch.randn(1, requires_grad = True, dtype = torch.float)
    b = torch.randn(1, requires_grad = True, dtype = torch.float)
    print(a, b)
    # 第二种方法
    a = torch.randn(1, requires_grad = True, dtype = torch.float).to(device)
    b = torch.randn(1, requires_grad = True, dtype = torch.float).to(device)
    print(a, b)
    # 第三种方法
    a = torch.randn(1, dtype = torch.float).to(device)
    b = torch.randn(1, dtype = torch.float).to(device)
    a.requires_grad_()
    b.requires_grad_()
    print(a, b)
    # 创建时即确定
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    print(a, b)
    
    # 5.自动梯度
    lr = 1e-1
    n_epochs = 1000
    
    for epoch in range(n_epochs):
        yhat = a + b*x_train_tensor
        error = y_train_tensor - yhat
        loss = (error**2).mean()
        
        # 不用自己手动计算梯度了
        loss.backward()
        # print(a.grad)
        # print(b.grad)
        
        # 更新参数，这时不需要自动计算梯度
        with torch.no_grad():
            a -= lr*a.grad
            b -= lr*b.grad
            
        # 将梯度置零，使过程继续
        a.grad.zero_()
        b.grad.zero_()
        
    print(a, b)
    
    # 6.动态计算图
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    yhat = a + b*x_train_tensor
    error = y_train_tensor - yhat
    loss = (error**2).mean()
    graph = make_dot(yhat)
    # graph.view("./output/yhat")
    
    # 7.优化
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    print(a, b)
    
    lr = 1e-1
    n_epochs = 1000
    
    optimizer = optim.SGD([a, b], lr = lr)
    for epoch in range(n_epochs):
        yhat = a + b*x_train_tensor
        error = y_train_tensor - yhat
        loss = (error**2).mean()
        
        # 不用自己手动计算梯度了
        loss.backward()
        
        # 也不用自己手动更新参数了
        optimizer.step()
        # 也不用手动将梯度归零
        optimizer.zero_grad()
        
    print(a, b)
    
    # 8.损失函数 loss
    # pytorch提供了很多损失函数计算方法
    # 还可以通过reduction参数来决定如何聚合单个神经节的损失。
    torch.manual_seed(42)
    a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
    print(a, b)
    
    lr = 1e-1
    n_epochs = 1000
    
    # 使用pytorch的损失函数
    loss_fn = nn.MSELoss(reduction='mean')
    
    optimizer = optim.SGD([a, b], lr = lr)
    for epoch in range(n_epochs):
        yhat = a + b*x_train_tensor
        # error = y_train_tensor - yhat
        # loss = (error**2).mean()
        # 不用自己算中间值了
        loss = loss_fn(y_train_tensor, yhat)
        
        # 不用自己手动计算梯度了
        loss.backward()
        
        # 也不用自己手动更新参数了
        optimizer.step()
        # 也不用手动将梯度归零
        optimizer.zero_grad()
        
    print(a, b)
    
    # 9.模型
    # 在pytorch中模型是继承自Module的一个类
    # 至少要实现__init__，初始化参数
    # 和forward，是实际计算过程，给定参数x，
    # 输出预测。
    # 使用model(x)来做出预测
    # 模型和数据应该在同一设备中
    class ManualLinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            """使用nn.Parameter使a,b成为模型真正的参数,可以通过parameters()获得参数列表，还可以通过state_dict()获得所有参数的当前值"""
            self.a = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
            self.b = nn.Parameter(torch.randn(1, requires_grad = True, dtype = torch.float))
            
        def forward(self, x):
            # 实际的计算过程
            return self.a + self.b*x
            
    # 使用模型
    
    torch.manual_seed(42)
    
    #创建模型并传到相关设备上
    model = ManualLinearRegression().to(device)
    # 输出模型参数状态
    print(model.state_dict())
    
    lr = 1e-1
    n_epochs = 1000
    
    loss_fn = nn.MSELoss(reduction = "mean")
    optimizer = optim.SGD(model.parameters(), lr = lr)
    
    for epoch in range(n_epochs):
        # 这里不是训练，只是开启训练模式
        # 因为有的模型会使用诸如Dropout等
        # 它们在训练阶段和评估阶段的行为不同
        model.train()
        
        # 不用手动计算了
        yhat = model(x_train_tensor)
        
        loss = loss_fn(y_train_tensor, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print(model.state_dict())
    
    #嵌套模型 nested models
    class LayerLinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    # 使用模型
    
    torch.manual_seed(42)
    
    #创建模型并传到相关设备上
    model = LayerLinearRegression().to(device)
    # 输出模型参数状态
    print(model.state_dict())
    
    lr = 1e-1
    n_epochs = 1000
    
    loss_fn = nn.MSELoss(reduction = "mean")
    optimizer = optim.SGD(model.parameters(), lr = lr)
    
    for epoch in range(n_epochs):
        # 这里不是训练，只是开启训练模式
        # 因为有的模型会使用诸如Dropout等
        # 它们在训练阶段和评估阶段的行为不同
        model.train()
        
        # 不用手动计算了
        yhat = model(x_train_tensor)
        
        loss = loss_fn(y_train_tensor, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    print(model.state_dict())
    
    # 序列模型Sequential Models
    # 为了不用新建一个类
    #对于前馈模型，前一层输出可以作为后层的输入
    model = nn.Sequential(nn.Linear(1, 1)).to(device)
    
    # 可以写一个函数封装固定的训练过程
    def make_train_step(model, loss_fn, optimizer):
        # 执行在循环中训练过程
        def train_step(x, y):
            # 设置训练模式
            model.train()
            # 预测
            yhat = model(x)
            # 计算损失
            loss = loss_fn(y, yhat)
            # 计算梯度
            loss.backward()
            # 更新参数，梯度置零
            optimizer.step()
            optimizer.zero_grad()
            # 返回损失值
            return loss.item()
        
        # 返回在训练循环中调用的函数
        return train_step
        
    torch.manual_seed(42)
    
    #创建模型并传到相关设备上
    model = LayerLinearRegression().to(device)
    # 输出模型参数状态
    print(model.state_dict())
    
    lr = 1e-1
    n_epochs = 1000
    
    loss_fn = nn.MSELoss(reduction = "mean")
    optimizer = optim.SGD(model.parameters(), lr = lr)
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    
    for epoch in range(n_epochs):
        loss = train_step(x_train_tensor, y_train_tensor)
        losses.append(loss)
        
    print(model.state_dict())
    
    # 10.数据集 dataset
    # 代表继承自Dataset的一个类
    # 可看成一个tuples列表,每个tuple代表一个(特征，标签)点
    # 数据很大时，建议在需要时再加载，用__get_item__
    class CustomDataset(Dataset):
        # 用csv文件或tensor输入
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor
            
        def __getitem__(self, index):
            return (self.x[index], self.y[index])
            
        def __len__(self):
            return len(self.x)
            
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    
    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])
    # 如果一个数据集是一对张量，可以用TensorDataset
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    print(train_data[0])
    # 别把所有训练数据都放到GPU里，太占显存了
    # 创建数据集的目的是可以使用DataLoader
    
    # 11.加载数据DataLoader
    # 对于大数据集，在训练中只加载一部分
    train_loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True)
    
    # 使用
    losses = []
    train_step = make_train_step(model, loss_fn, optimizer)
    
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            
    print(model.state_dict())
    # 随机划分训练_验证集
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    
    dataset = TensorDataset(x_tensor, y_tensor)
    
    train_dataset, val_dataset = Data.dataset.random_split(dataset, [80, 20])
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = 16)
    val_loader = DataLoader(dataset = val_dataset, batch_size = 20)
    
    # 12.评估
    losses = []
    val_losses = []
    train_step = make_train_step(model, loss_fn, optimizer)
    
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                # 将模型置为评估阶段
                model.eval()
                
                yhat = model(x_val)
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())
                
    print(model.state_dict())
    

if __name__ == "__main__":
    # nn_numpy()
    # nn_pytorch()
    # nn_torch_nn()
    # pytorch_class()
    # print(torch.__version__)
    # print(torch.version.cuda)
    new_try()