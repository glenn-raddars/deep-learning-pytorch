import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])
w_list = []# 横坐标
loss_list = []#纵坐标

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        """实例化一个Linear类"""
        self.Linear = torch.nn.Linear(1, 1)#这两个参数一个是输入维度，一个是输出维度，就是数据集的特征个数，
        #因为这两个数据集的列都是1，所以是1,还有一个参数bias=,这个是是否计算偏置，默认为TRUE

    def forward(self,x):
        """必须要有这么一个forward函数，因为在父类里，这个函数在__call__中，是可以随着对象被调用的，这里是重写覆盖"""
        y_pred = self.Linear(x)#这个实例是可以调用的，因为在父类中将Torch.nn.Linear设置成了可调用的类
        #可调用类标志为__call__
        return y_pred

model = LinearModel()#实例化线性回归模型
"""实例化损失函数模型"""
criterion = torch.nn.MSELoss(size_average=False)#不用将loss规范化，即不除以样本个数
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
    print("w权重为: ",model.Linear.weight.item())
    print("b偏置为: ",model.Linear.bias.item())

    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print("最终预测值y：",y_test.data.item())

    plt.plot(w_list,loss_list)
    plt.xlabel("w")
    plt.ylabel("loss")
    plt.show()
