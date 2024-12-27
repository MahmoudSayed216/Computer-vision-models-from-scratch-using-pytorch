import torch

x1 = torch.tensor([1,2,3])
x2 = torch.tensor([1,4,3])

print(sum(x1==x2)/2)

x1.tolist()