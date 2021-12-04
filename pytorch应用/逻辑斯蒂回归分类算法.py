import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])
w_list = []# 横坐标
loss_list = []#纵坐标

"""逻辑斯蒂函数，就是σ = 1 / (1 + e^-x),这个函数可以把值映射到 [0,1] 这个区间，sigmoid函数是一组映射函数，因为逻辑斯蒂是这个里面最具有代表性的函数
因此，将sigmoid函数就视为逻辑斯蒂函数。
本次分类是在线性回归的基础上，每周学习时间在多少以上期末考试能及格，所以分类结果只有0或1是个二分类问题，最终输出的是映射结果，及及格的概率"""

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        """实例化一个Linear类"""
        self.Linear = torch.nn.Linear(1, 1)#这两个参数一个是输入维度，一个是输出维度，就是数据集的特征个数，
        #因为这两个数据集的列都是1，所以是1,还有一个参数bias=,这个是是否计算偏置，默认为TRUE

    def forward(self,x):
        """必须要有这么一个forward函数，因为在父类里，这个函数在__call__中，是可以随着对象被调用的，这里是重写覆盖"""
        y_pred = F.sigmoid(self.Linear(x)) #这个实例是可以调用的，因为在父类中将Torch.nn.Linear设置成了可调用的类，并调用sigmoid函数映射到[0,1]区间
        #可调用类标志为__call__
        return y_pred

model = LogisticRegression()#实例化线性回归模型
"""实例化损失函数模型"""
criterion = torch.nn.BCELoss(size_average=False)#不用将loss规范化，即不除以样本个数l,这里用的是二分类算法的损失函数，为交叉熵，越小越好（取得负值）
#讲交叉熵的https://zhuanlan.zhihu.com/p/35709485
"""实例化优化器"""
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)#第一个参数是告诉优化器我要计算哪些参数（优化，更新）在线性回归里就是权重w
#和偏置b，model.parameters()会检查并调用Linear.parameters去里面找参数，就会找到w和b，lr是学习率

if __name__ == "__main__":
    for epoch in range(10000):
        y_pred = model(x_data)#预测值
        loss = criterion(y_pred,y_data)#损失函数
        print("这是第%d轮,损失是；"%epoch,loss.item())

        loss_list.append(loss.item())

        """计算梯度之前先把以前的梯度清零"""
        optimizer.zero_grad()
        loss.backward()#开始回退计算每个张量的梯度
        optimizer.step()#更新权重
        w_list.append(model.Linear.weight.item())

    """整个跑完之后"""
    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print("最终预测值y：",y_test.data.item())

    """用这个模型来绘制一整个一周学习时间和考试及格率预测图"""
    x = np.linspace(0, 10,200)#详细讲解此函数https://blog.csdn.net/Asher117/article/details/87855493
    x_test = torch.Tensor(x).view(200,1)#改变Tensor的尺寸
    y_test = model(x_test)
    y = y_test.data.numpy()#将预测值转化为数组

    plt.plot(x,y)
    plt.plot([0,10],[0.5,0.5],c='r')
    plt.xlabel("Hours")
    plt.ylabel("Probability of Pass")
    plt.grid()#画出坐标格，显示网格线
    plt.show()
    
