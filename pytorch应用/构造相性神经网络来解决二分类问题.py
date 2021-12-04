import torch
import matplotlib.pyplot as plt
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        """由于构建神经网络，需要多层线性回归，中间增加激活函数将线性转变为非线性"""
        """输入数据集x特征8维，先转化成6维，再转化成4维，再转化成1维，中间都用sigmoid函数做非线性变换"""
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        """激活函数"""
        self.activity = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        """每次都用x进行迭代"""
        x = self.activity(self.linear1(x))
        x = self.activity(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()#实例化
criterion = torch.nn.BCELoss(size_average=False)#实例化损失函数，交叉熵，不规范化
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)#实例化优化器

if __name__ == "__main__":
    """准备糖尿病患者的数据集"""
    xy = np.loadtxt("C:\python\python3.9-Mindspore-深度学习\课件和数据集\PyTorch深度学习实践\diabetes.csv.gz",dtype=np.float32,delimiter=',')
    """划分数据集"""
    x_data = torch.from_numpy(xy[:,:-1])#简单说一下，就是torch.from_numpy()方法把数组转换成张量，
    #且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。这里是踢出最后一列，剩下的作为数据集
    y_data = torch.from_numpy(xy[:,[-1]])#只拿最后一列作为数据集.[-1]是为了使张量中的data是二维数组
    
    for epoch in range(10000):
        y_pred = model(x_data)
        #print(y_pred.data)
        loss = criterion(y_pred,y_data)
        print("低%d次训练的损失为: "%epoch,loss.item())

        #计算梯度
        optimizer.zero_grad()
        loss.backward()

        #更新梯度
        optimizer.step()




