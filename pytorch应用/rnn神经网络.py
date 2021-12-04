import torch
import torch.optim as optim

"""RNN循环神经网络主要用来处理跟时间，或者样本之间存在逻辑关系的数据集，比如天气，我们要通过近几天的气温变化推出后面的天气。
比如自然语言，我们要根据前面那个人说过的话来推出后面他需要说什么"""
"""循环神经网络其实主要是一个线性层一直不停的反复训练，每次都训练那一个线性层，所以叫做循环神经网络"""

input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e','h','l','o']#意思是 e对应0 h对应1...以此类推
"""训练一个模型，可以吧hello转化为ohlol"""
x_data = [1,0,2,2,3]#hello
y_data = [3,1,2,3,2]#ohlol

#独热向量索引表
one_hot_lookup = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]#只有四个字母，所以有4列，第一行对应的是哪个字母，字母对应的列就是1，其余列就是0
#讲x_data转化成独热向量
x_one_hot = [one_hot_lookup[x] for x in x_data]
#print(x_one_hot)
inputs1 = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)#序列长度seq，batch_size，输入维度

labels1 = torch.LongTensor(y_data).view(-1,1)#输出维度为1，所以开头的序列长度为5
labels2 = torch.LongTensor(y_data)#保留连在一起的一整串


class Model1(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model1, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        """最后他会把inputs的维度转化为hidden的维度，然后跟hidden相加并且送到下一次RNN中去运算"""
        self.rnncell = torch.nn.RNNCell(input_size = self.input_size, hidden_size = self.hidden_size)

    def forward(self,inputs,hidden):
        hidden = self.rnncell(inputs,hidden)
        return hidden
    """初始化hidden的函数"""
    def init_hidden(self):
         return torch.zeros(self.batch_size,self.hidden_size)

"""使用RNN搭建RNN神经网络，这属于是一次性把所有的输入序列seq处理完"""
class Model2(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers = 1):
        super(Model2, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.run = torch.nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size,num_layers=num_layers)
        #num_layers是神经网络层数

    def forward(self,inputs):
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out,_ = self.run(inputs,hidden)#同时有匹配序列和最后一层的hidden输出，只要匹配序列
        return out.view(-1,self.hidden_size)#把他变成 seq*batch_size，hidden_size的格式

model1 = Model1(input_size, hidden_size, batch_size)
model2 = Model2(input_size, hidden_size, batch_size)
device = torch.device("cuda:0")
#model1.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(),lr=0.1)
optimizer2 = optim.Adam(model2.parameters(),lr=0.1)

if __name__ == "__main__":
    temp = int(input("input:"))

    if temp == 1:
        print("ok")
        for epoch in range(15):
            loss = 0.0
            hidden = model1.init_hidden()
            #hidden.to(device)
            optimizer1.zero_grad()
            print("预测的字符串：",end=' ')
            for in_put,label in zip(inputs1,labels1):#因为inputs是seq，batch_size，input_size，实际上in_put拿到的就是x1,x2,x3这种序列，然后
                #然后batch_size又把这些序列分成batch_size对应的份数
            # in_put.to(device)
            # label.to(device)
                #print("in_put:\n",in_put)
                #print("label:\n",label)
                hidden = model1(in_put,hidden)
                #print("hidden:\n",hidden)
                #hidden.to(device)
                loss += criterion(hidden,label)
                #loss.to(device)
                #拿到最大的可能字母序列
                _,idx = hidden.max(dim = 1)
                #idx.to(device)
                print(idx2char[idx.item()],end=' ')

            loss.backward()
            optimizer1.step()
            print(",Epoch[%d/15] loss = %.4lf"%(epoch+1,loss.item()))

    else:
        for epoch in range(15):
            loss = 0.0
            optimizer2.zero_grad()
            outputs = model2(inputs1)
            #print("outputs:\n",outputs)
            #print("labels2:\n",labels2) 
            loss = criterion(outputs,labels2)
            loss.backward()
            optimizer2.step()

            _,idx = outputs.max(dim = 1)
            idx = idx.data.numpy()
            #print(idx)

            print("Predicted: "," ".join([idx2char[x] for x in idx]),end=' ')
            print(",Epoch[%d/15] loss = %.4lf"%(epoch+1,loss.item()))
            
                    



