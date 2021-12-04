import torch
import matplotlib.pyplot as plt
#什么是BP算法？https://zhuanlan.zhihu.com/p/45190898

"""这里首先是想写一个极其简单的的线性回归，不带激活函数 y = w*x"""
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
#构造一个Tensor张量，为初始权重
w = torch.tensor([1.0])#给初始权重赋值为1.0
w.requires_grad = True #表明需要对w这个张量求梯度，一个张量是由数值和他的梯度构成的（最终求得的损失函数MSE对w的导数）

w_list = []
loss_list = []

def forward(x):#前馈函数（由前馈函数一层一层的推导出最后的回归函数和损失函数）
    return x*w#现在x也是一个tensor，并且也会对 x*w 求梯度，因为里面含有w

def loss(x,y):#损失函数
    y_pred = forward(x)
    return (y_pred - y)**2

if __name__ == "__main__":
    print("前馈函数的结果得到的预测值",4,forward(4).item())

    for epoch in range(100):
        for x,y in zip(x_data ,y_data):
            l = loss(x, y)
            l.backward()
            print("\tgrad:",x,y,w.grad.item(),w.data.item())#因为这个里面梯度也是一个张量，所以要解包
            w.data = w.data - 0.05*w.grad.data#必须要用数据，否则就又会建立新的张量，新的计算图
            w.grad.data.zero_()#需要吧w的梯度清零，不然会把几次测试的梯度一直累加

        print("progross",epoch,l.item())
        w_list.append(w.data.item())
        loss_list.append(l.item())

    #整个训练完毕后的预测值
    print("前馈函数的结果得到的预测值",4,forward(4).item())
    plt.plot(w_list,loss_list)
    plt.ylabel("loss")
    plt.xlabel("w")
    plt.show()