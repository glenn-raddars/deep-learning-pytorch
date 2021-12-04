from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
"""将数据集用mini-batch处理"""

"""首先构造处理数据集的类"""
class DiabetesDataset(Dataset):#继承自Dataset这个虚基类
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,dtype=np.float32,delimiter=',')#单纯的变量，不是属性
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

"""实例化数据集对象"""
dataset = DiabetesDataset("C:\python\python3.9-Mindspore-深度学习\课件和数据集\PyTorch深度学习实践\diabetes.csv.gz")
"""实例化mini-batch的训练器"""
train_loader = DataLoader(dataset = dataset,batch_size=32,shuffle=True,num_workers=2)
#dataset就是数据集，batch_size是一次mini-batch的大小，就是他会把数据集按照32个数据一组划分，一次batch就要训练 总个数/32次，
#shuffle为是否打乱原有的数据顺序，num_workers是有几个进程一起读取数据（几个核）

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
criterion = torch.nn.BCELoss(reduction="sum")#实例化损失函数，交叉熵，不规范化,这里reduce为sum就是不规范化，mean就是规范化
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)#实例化优化器

if __name__ == "__main__":
    for epoch in range(100):
        for i,data in enumerate(train_loader, 0):
            inputs,labels = data#train_loader吧x_data和y_data合并了,现在在把他拆开

            y_pred = model(inputs)
            loss = criterion(y_pred,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #到这里一次batch就完整的跑完了，然后再跑100次
        print("第%d次训练损失为："%epoch,loss.item())