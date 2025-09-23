import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

a = torch.FloatTensor(3, 2)
print(a)

# Initialize a tensor with zeros for good practice
print(torch.zeros(3, 4))


# Tensor modification method
print(a.zero_())

# Create tensor from Python iterable
n = np.zeros(shape=(3, 2))
print(n)

b = torch.tensor(n)  # creates a double (64-bit-float) by default
print(b)

# Specify the dtype from numpy
n = np.zeros(shape=(3, 2), dtype=np.float16)
b = torch.tensor(n)
print(b)

# Specify the dtype from torch
n = np.zeros(shape=(3, 2))
b = torch.tensor(n, dtype=torch.float16)
print(b)

a = torch.FloatTensor([2, 3])
print(a)

if torch.cuda.is_available():
    ca = a.to("cuda")  # creates gpu-enabled tensor if available
else:
    ca = a
print(ca)


print(a + 1)
print(ca + 1)
print(ca.device)

# Experimenting with requires_grad
v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])
v_sum = v1 + v2
print(v_sum)

v_res = (v_sum * 2).sum()
print(v_res)

# Check attributes of the tensors
print(v1.is_leaf, v2.is_leaf)

print(v_sum.is_leaf, v_res.is_leaf)

print(v1.requires_grad)

print(v2.requires_grad)

print(v_sum.requires_grad)

print(v_res.requires_grad)
v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])
v_sum = v1 + v2
print(v_sum)

v_res = (v_sum * 2).sum()
print(v_res)

# Calculate the gradients
v_res.backward()
print(v1.grad)

# torch.nn package - quick view

# Create a randomly initialized feed-forward layer with 2 inputs and 5 outputs and applied it to our tensor.
l = nn.Linear(2, 5)
v = torch.FloatTensor([1, 2])
print(l(v))

# Experimenting with Sequential class pipeline
s = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1),
)

print(s)

out = s(torch.FloatTensor([[1, 2]]))
print(out)


# Creating a Custom Module
class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super().__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.pipe(x)


# TENSORBOARD


# Using custom module:

if __name__ == "__main__":
    writer = SummaryWriter()
    funcs = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    for angle in range(-360, 360):
        angle_rad = angle * math.pi / 180
        for name, fun in funcs.items():
            val = fun(angle_rad)
            writer.add_scalar(name, val, angle)
    writer.close()

    net = OurModule(num_inputs=2, num_classes=3)
    print(net)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(out)
    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to("cuda"))
