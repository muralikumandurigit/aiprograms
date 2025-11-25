import torch
import torch.nn as nn

t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t)

x = torch.tensor([[3.4, 0.1], [1.2, 4.3]])
print(x)

y = torch.zeros(4, 3)
print(y)

z = torch.ones(2, 5)
print(z)

a = torch.randn(3, 3)
print(a)

b = torch.randn(3, 2)
print(b)

c = torch.eye(2)
print(c)


layer1 = nn.Linear(3, 2)
input1 = torch.randn(4, 3)
output1 = layer1(input1)
print(input1)
print(layer1.weight)
print(layer1.bias)
print(output1)


print("Exercise")

x = torch.randn(4, 3)
y = 3*x + 2
print(x)
print(3*x)
print(y)