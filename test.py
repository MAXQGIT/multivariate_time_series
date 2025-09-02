import torch
import torch.nn as nn

input = torch.arange(8, dtype=torch.float).reshape(1, 2, 4)
print(input)
m = nn.ReplicationPad1d((0, 2))
a = m(input)
print(a)
a = a.unfold(dimension=-1, size=4, step=2)
print(a)




