import torch
a = torch.Tensor([[1,2],[3,4]])
a.requires_grad_(True)
b = a*3
print(a.requires_grad)
b.backward(torch.ones([2,2]))
print(a.grad)

