import torch
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import gzip
import csv
import torch.optim as optim

"""RNN神经网络分类器，用来判断姓名属于哪个国家"""
"""首先拿到，并处理数据集"""
BATCH_SIZE = 256
HIDDLE_SIZE = 100
N_LAYER = 2
N_EPOCHS = 100
N_CHARS = 128#ASCII码字符集128个

class NameDataset(Dataset):
    def __init__(self,is_train_set = True):
        super(NameDataset, self).__init__()
        """读取数据集的路径"""
        filename = "C:\\python\\python3.9-Mindspore-深度学习\\课件和数据集\\PyTorch深度学习实践\\names_train.csv.gz" if is_train_set else "C:\python\python3.9-Mindspore-深度学习\课件和数据集\PyTorch深度学习实践\\names_test.csv.gz"
        with gzip.open(filename,'rt') as f:#rt模式是按照文本读取此压缩文件
            reader = csv.reader(f)#读取csv文件中的每一行，并且作为字符串列表返回
            rows = list(reader)
        self.names = [row[0] for row in rows]#拿取每一行的第零列作为姓名
        self.len = len(self.names)#返回整个训练数据集的长度
        self.countries = [row[1] for row in rows]#拿取每一行的第1列作为国家
        self.country_list = list(sorted(set(self.countries)))#把重复的国家去除，并按照首字母排序
        self.country_dic = self.getCountrydict()#将国家字典化
        self.country_num = len(self.country_list)#国家数量

    def __getitem__(self, index):
        return self.names[index],self.country_dic[self.countries[index]]#将来返回的是姓名，和国家对应的索引

    def __len__(self):
        return self.len

    def getCountrydict(self):
        country_dict = dict()
        for idx,country_name in enumerate(self.country_list):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self,index):
        return self.country_list[index]
    
    def getCountriesNum(self):
        return self.country_num

trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset,batch_size = BATCH_SIZE,shuffle = True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)
N_COUNTRY = testset.getCountriesNum()
#print(N_COUNTRY)

class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers = 1,bidirectional = True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        #嵌入层,将高维向量转换成低维稠密向量
        self.embedding = torch.nn.Embedding(input_size, hidden_size)#输入的维度是（seq,batch_size），输出的维度是(seq,batch_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size,n_layers,bidirectional=bidirectional)#bidirectional决定是否再从尾往头算一遍
        self.fc = torch.nn.Linear(hidden_size*self.n_direction, output_size)#如果从头往回算了一遍，那么输出结果是两次结果相拼接而成，所以输入维度是原来的两倍

    def init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers*self.n_direction,batch_size,self.hidden_size)
        return hidden
    
    def forward(self,inputs,seq_len):
        #将input转置，之前是batch_size*seq 现在是 seq*batch_size
        inputs = inputs.t()
        batch_size = inputs.size(1)
        #构建初始隐层
        hidden = self.init_hidden(batch_size)
        embedding = self.embedding(inputs)#这个时候维度就是 seq*batch_size*hidden_size
        
        #将embedding层的输出转化为更好的GRU输入
        """因为在做处理的时候会按照长度最长的人名对所有人名进行扩充，把所有人名字变得一样长，但是这样会增加许多无谓的计算
        所以使用 pack_padded_sequence 来将输入进行转变，将来就不用计算扩充的0"""
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding,seq_len)

        output,hidden = self.gru(gru_input,hidden)
        #print(hidden.size())
        if self.n_direction == 2:
            #输出的 hidden有两个维度，要合并
            #print(hidden[-1].size())
            #print(hidden[-2].size())
            hidden_cat = torch.cat([hidden[-1],hidden[-2]],dim = 1)
        else:
            hidden_cat = hidden[-1]
        #print(hidden_cat.shape)
        fc_output = self.fc(hidden_cat)
        return fc_output

def name2list(name):
    arr = [ord(c) for c in name]#把一个名字中的每一个字母转换成ascll
    return arr,len(arr)#返回元祖

def make_tensors(names,countries):
    squence_and_length = [name2list(name) for name in names]#那整个输入转化成ascll码加单词长度
    name_sequences = [s1[0] for s1 in squence_and_length]#单独把名字拿出来
    seq_lengths = torch.LongTensor([s1[1] for s1 in squence_and_length])#单独吧名字长度拿出来
    countries = countries.long()#最后做损失只能是长整型

    #开始对名字进行补全
    seq_tensor = torch.zeros(len(name_sequences),seq_lengths.max()).long()
    #相当于创造一个全零的模板，然后复制粘贴名字
    for idx,(seq,seq_length) in enumerate(zip(name_sequences,seq_lengths),0):
        seq_tensor[idx,:seq_length] = torch.LongTensor(seq)

    #为了使用pack_padded_sequence函数，对输入数据进行降序排序
    seq_lengths,perm_idx = seq_lengths.sort(dim = 0,descending = True)#他返回排好的张量，和在原张量上的索引值
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return seq_tensor,seq_lengths,countries

classifier = RNNClassifier(N_CHARS, HIDDLE_SIZE, N_COUNTRY,N_LAYER)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(),lr=0.001)

def trainModel():
    total_loss = 0.0
    for i,(names,countries) in enumerate(trainloader,1):
        inputs,seq_lengths,targets = make_tensors(names, countries)#把输入，序列长度，转化目标提取出来
        output = classifier(inputs,seq_lengths)#这个地方就已经映射到跟国家一样的维度空间了

        optimizer.zero_grad()
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()

        total_loss += loss
        if i%10 == 0:
            print("loss: %.5lf"%(total_loss/i))
        
    #return total_loss

def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model...")
    with torch.no_grad():
        for i,(names,countries) in enumerate(testloader,1):
            inputs,seq_lengths,targets = make_tensors(names, countries)

            output = classifier(inputs,seq_lengths)
           # print("output:\n",output)
            #print("output.szie:\n",output.size())
            _,pred = output.max(dim = 1)
            #print("pred:\n",pred)
            #print("pred.size:\n",pred.size())
            #print("target.size:\n",targets.size())
            correct += pred.eq(targets.view_as(pred)).sum().item()

        print("准确率为: %.5lf"%(100*correct/total))
    
    return correct/total


if __name__ == "__main__":
    #dataset1 = NameDataset(is_train_set=True)
    #print("countries\n",dataset1.countries)
    #print('dict:\n',dataset1.country_dic)
    #print(trainloader.size)
    acc_list = []
    epoch_list = []
    print("training for %d epochs..."%N_EPOCHS)
    for epoch in range(1,N_EPOCHS+1):
        print("第%d轮训练:"%epoch)
        trainModel()
        accc = testModel()

        epoch_list.append(epoch)
        acc_list.append(accc)

    plt.plot(epoch_list,acc_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    

