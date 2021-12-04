import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
"""之前linear都是全连接神经网络,这次做的是卷积神经网络"""

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

"""构建卷积神经网路"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #对输入图像做第一层卷积
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5)#这里bais会加一个默认偏置为TRUE,这里把图像通道数从1维变成10维
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5)#通道数从10维变成20维
        #设置池化层，就是将一个通道的图像分割，在那个区域中找出像素值最高的像素点代替整个区域
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)#分割成2*2的区域，用来减小图片大小，但不会影响图片维度
        self.fc = torch.nn.Linear(320, 10)#做线性层的激活映射到10个分类的维度

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))#需要relu函数对结果做非线性激活
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)#64实际上就是batch_size，一次输入64张图片，剩下就自动计算一张图片几个像素点(实际上就是320)
        #print(x.size(1))
        x = self.fc(x)
        return x

model = Net()
"""来试试使用显卡"""
device = torch.device("cuda:0")
#把模型放到显卡上跑
model.to(device)
criterion = torch.nn.CrossEntropyLoss()#也是交叉熵损失，但是是多分类问题的
optimizer = optim.SGD(model.parameters(), lr = 0.01,momentum=0.5)#这里momentum是在设置冲量，用来冲破局部最小值

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader):
        inputs,targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            print("[%d, %5d] loss: %.3lf"%(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs,targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            """用max函数查找每一行的最大值，那个就是模型预测的分类，会返回最大值和最大值的下标"""
            _,predicts = torch.max(outputs,dim=1)#dem=1就是按照列这个维度找一行中哪一列的值最大

            total += targets.size(0)#计算整个结果数量
            correct += (predicts == targets).sum().item() #计算正确结果的数量
        
        print("准确率为:%d %%"%(100*correct/total))

if __name__ == "__main__":
    for epoch in range(100):
        train(epoch)
        test()


