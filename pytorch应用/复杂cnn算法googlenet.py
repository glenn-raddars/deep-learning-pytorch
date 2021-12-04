import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

"""Googlenet神经网络"""

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

"""构建GoogleNet神经网络"""
class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):#in_channels是输入维度，及输入通道数
        super(InceptionA, self).__init__()
        """第一层通道"""
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size = 1)#卷积核大小为1*1的卷积，输出通道数为16
        """第二层通道,5*5的卷积通道"""
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size = 1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size = 5,padding=2)#先做1*1卷积，再做5*5卷积最后输出24个通道,1*1卷积有助于降低计算量
        #因为要保证每个通道最后输出的张量每个图像的宽度和高度一致,所以padding要设置为2，在卷积的时候在周围填充2圈不改变原图像大小
        """第三层通道，3*3卷积通道"""
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, out_channels = 16, kernel_size = 1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3,padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size = 3,padding=1)
        """第四层通道，池化层"""
        self.branchpool = torch.nn.Conv2d(in_channels, 24, kernel_size = 1)

    def forward(self,x):
        """第一层通道输出,16通道输出"""
        branch1x1 = self.branch1x1(x)
        """第二层通道的输出,24通道输出"""
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        """第三层通道的输出，24通道输出"""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        """第四层通道的输出，24通道输出"""
        branchpool = F.avg_pool2d(x,kernel_size = 3,padding = 1,stride = 1)#讲这个函数的https://blog.csdn.net/u013066730/article/details/102553073
        #上面平均池化函数不会改变channel的数量
        branchpool = self.branchpool(branchpool)

        """将上述所有通道的输出按照channel合并"""
        outputs = [branch1x1,branch5x5,branch3x3,branchpool]
        return torch.cat(outputs,dim = 1)#cat可以把张量按照特定维度结合,dim=1就是按照通道结合，因为图像的张量都是[batch_size,channel,height,width],
        #所以dim = 1就是channels

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size =5)
        self.conv2 = torch.nn.Conv2d(88,20 , kernel_size = 5)
        """使用googleNet网络"""
        self.incep1 = InceptionA(in_channels = 10)#跟conv1对应
        self.incep2 = InceptionA(in_channels = 20)#跟conv2对应
        """最大池化层"""
        self.mp = torch.nn.MaxPool2d(kernel_size = 2)
        """全连接层"""
        self.linear = torch.nn.Linear(1408, 10)

    def forward(self,x):
        in_size = x.size(0)#实际上就是batch的数量
        x = F.relu(self.mp(self.conv1(x)))#输出维度10
        x = self.incep1(x)#输出维度24+24+24+16 = 88

        x = F.relu(self.mp(self.conv2(x)))#输出维度20
        x = self.incep2(x)#输出维度88
        x = x.view(in_size,-1)
        #print(x.size(1))
        #最后的全连接层
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
        loss = criterion(outputs,targets)

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