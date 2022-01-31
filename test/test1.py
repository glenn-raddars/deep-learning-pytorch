import torch
a = torch.tensor([[1,2,3],[3,4,5]])
a = a.unsqueeze(0)
a = a.transpose(0,1)
a = a.transpose(1,2)
print(a.size())
a = a.permute(2,0,1)
print(a.shape)