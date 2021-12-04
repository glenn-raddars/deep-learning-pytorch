import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

"""构建数据集的转换器，transforms.ToTensor()是将拿到的数据集转化为张量，数据集里的图片是28*28的像素，经过转换，会变成1*28*28的张量（1是通道数量），
transforms.Normalize((0.1307,), (0.3081))是把数据集变成均值为0.1307，标准差为0.3081的正态分布"""
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081))])
"""下载数据集"""
train_dataset = datasets.MNIST(root="C:\\python\\python3.9-Mindspore-深度学习\\课件和数据集\\minist\\",
train=True,transform=transform,download=False)#train为True下载训练集，反之下载数据集，download确定是否下载，transform设置数据集的转换器
test_dataset = datasets.MNIST(root="C:\\python\\python3.9-Mindspore-深度学习\\课件和数据集\\minist\\",
transform=transform,train = False,download=False)
"""对数据集做mini-batch处理"""
train_loader = DataLoader(dataset = train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset = test_dataset,batch_size=64,shuffle=False)

"""构建神经网络"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #5层线性回归
        self.linear11 = torch.nn.Linear(784, 512)#一开始的输入维度是28*28 = 784
        self.linear12 = torch.nn.Linear(512, 256)
        self.linear13 = torch.nn.Linear(256, 128)
        self.linear14 = torch.nn.Linear(128, 64)
        self.linear15 = torch.nn.Linear(64, 10)#最后有10各类别

    def forward(self,x):
        """数据集里的图片实际上是64*1*28*28，64就是这一批batch中图片的数量，而要把它转化成二维矩阵，一张图片有784的像素点，
        而样本个数会有-1自动根据 总数/784 自动算出，就是64张图片"""
        x = x.view(-1,784)
        x = F.relu(self.linear11(x))
        x = F.relu(self.linear12(x))
        x = F.relu(self.linear13(x))
        x = F.relu(self.linear14(x))
        return self.linear15(x)#最后一层神经网络不做激活操作，因为之后所用到的损失函数会自动用softmax中的方法激活

model = Net()
"""https://blog.csdn.net/weixin_38314865/article/details/104311969
是讲这个损失函数的，其中 e^xi / Σe^xi 就是softmax激活函数，也就是说这个损失函数自带这个激活函数，将其映射到[0,1]区间"""
criterion = torch.nn.CrossEntropyLoss()#也是交叉熵损失，但是是多分类问题的
optimizer = optim.SGD(model.parameters(), lr = 0.01,momentum=0.5)#这里momentum是在设置冲量，用来冲破局部最小值

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,labels = data

        y_pred = model(inputs)
        loss = criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        """每训练300轮，输出一次损失"""
        if (batch_idx % 300) == 299:
            print("[%d,%5d] loss: %.3lf"%(epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():#以下张量不要计算梯度
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            """用max函数查找每一行的最大值，那个就是模型预测的分类，会返回最大值和最大值的下标"""
            _,predicts = torch.max(outputs,dim=1)#dem=1就是按照列这个维度找一行中哪一列的值最大

            total += labels.size(0)#计算整个结果数量
            correct += (predicts == labels).sum().item() #计算正确结果的数量

        print("准确率为:%d %%"%(100*correct/total))

if __name__ == "__main__":
    for epoch in range(100):
        train(epoch)
        test()