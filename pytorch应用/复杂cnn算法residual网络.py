import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

"""构建数据集的转换器，transforms.ToTensor()是将拿到的数据集转化为张量，数据集里的图片是28*28的像素，经过转换，会变成1*28*28的张量（1是通道数量），
transforms.Normalize((0.1307,), (0.3081))是把数据集变成均值为0.1307，标准差为0.3081的正态分布"""
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081))])
# transform = transforms.Compose([transforms.ToTensor()])
"""下载数据集"""
train_dataset = datasets.MNIST(root="C:\\python\\python3.9-Mindspore-深度学习\\课件和数据集\\minist",
train=True,transform=transform,download=False)#train为True下载训练集，反之下载数据集，download确定是否下载，transform设置数据集的转换器
test_dataset = datasets.MNIST(root="C:\\python\\python3.9-Mindspore-深度学习\\课件和数据集\\minist",
transform=transform,train = False,download=False)
"""对数据集做mini-batch处理"""
# train_data,train_label = train_dataset[0]
# train_data = train_data.resize(28,28)
# print(train_label)
# plt.imshow(train_data)
# plt.show()

train_loader = DataLoader(dataset = train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset = test_dataset,batch_size=64,shuffle=False)

"""residual是专门用来解决梯度下降到0导致无法继续下降的问题的，就是在几层训练之后在加上原来未训练的数据集，这样求导的时候始终有个+1，导致梯度不可能为零"""
class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size = 3,padding=1)#通道数及输入输出维度保持不变，才能和原始的数据集相加
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size = 3,padding=1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = torch.nn.Conv2d(1, 16, kernel_size = 5)
        self.Conv2 = torch.nn.Conv2d(16, 32, kernel_size = 5)

        """最大池化"""
        self.mp = torch.nn.MaxPool2d(kernel_size = 2)

        """residual层"""
        self.resblock1 = ResidualBlock(16)
        self.resblock2 = ResidualBlock(32)

        """全连接层"""
        self.linear = torch.nn.Linear(512, 10)

    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.Conv1(x)))
        x = self.resblock1(x)
        x = F.relu(self.mp(self.Conv2(x)))
        x = self.resblock2(x)
        x = x.view(in_size,-1)
        #print(x.size(1))

        return self.linear(x)

model = Net()
device = torch.device("cuda:0")
#把模型放到显卡上跑
model.to(device)
criterion = torch.nn.CrossEntropyLoss()#也是交叉熵损失，但是是多分类问题的
optimizer = optim.SGD(model.parameters(), lr = 0.01,momentum=0.5)#这里momentum是在设置冲量，用来冲破局部最小值

x_data = []
y_data = []

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader):
        inputs,targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        outputs.to(device)
        loss = criterion(outputs,targets)
        loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            print("[%d, %5d] loss: %.3lf"%(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
    x_data.append(epoch)

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs,targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs.to(device)

            """用max函数查找每一行的最大值，那个就是模型预测的分类，会返回最大值和最大值的下标"""
            _,predicts = torch.max(outputs,dim=1)#dem=1就是按照列这个维度找一行中哪一列的值最大

            total += targets.size(0)#计算整个结果数量
            correct += (predicts == targets).sum().item() #计算正确结果的数量
        
        print("准确率为:%d %%"%(100*correct/total))
        y_data.append((100*correct/total))

if __name__ == "__main__":
    for epoch in range(100):
        train(epoch)
        test()
    
    #print(x_data)
    #print(y_data)
    plt.plot(x_data,y_data)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()