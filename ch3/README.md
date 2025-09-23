# Deep Learning with PyTorch

## Tensor Data Types in PyTorch:

- (4) Float Types: 16-bit (float16 & bfloat16), 32-bit, 64-bit.

  - float16 has more bits for precision
  - bfloat16 has larger exponent part

- (3) Complex Types: 32-bit, 64-bit, 128-bit

- (5) Integer Types: 8-bit signed, 8-bit unsigned, 16-bit signed, 32-bit signed, 64-bit signed

- Boolean Type:

## 3 Ways to create a Tensor in PyTorch:

1. calling a constructor of the required type
2. ask PyTorch to create a tensor with specific data (ex: use torch.zeros())
3. convert a NumPy array or Python list into a tensor

```python
import torch
import numpy as np
a = torch.FloatTensor(3,2)
a
```

## Types of operation for tensors:

1. Inplace - have an underscore appended to their name and operate on the tensor's content
2. Functional - creates a copy of the tensor with the performed modification, leaving the original tensor untouched

## Create Tensor from Python iterable

```python
n = np.zeros(shape=(3,2))
print(n)

b = torch.tensor(n)
print(b)
```

> Torch creates 64-bit tensors by default which adds memory and overhead and usually 16-bit float type is more than enough.

## CPU vs GPU Tensors in PyTorch

Every tensor type mentioned above is for CPU and has a GPU equivalent.

CPU tensor = torch.FloatTensor (32-bit float tensor on CPU)
GPU tensor = torch.cuda.FloatTensor (32-bit float tensor on GPU)

# Convert CPU tensor to GPU tensor

To convert from a CPU tensor to a GPU tensor use the `.to(device)` method which creates a copy of the tensor to the specified device.

# Access the device the tensor was created on

To return the device that the tensor was created on use the `.device` property

```python
a = torch.FloatTensor([2,3])
print(a)

ca = a.to('cuda')
print(ca)


print(a+1)
print(ca+1)
print(ca.device)
```

# Automatic Gradient Calculation

What can make a fundamental difference is how your gradients are calculated.

1. `Static Graph`: you need to define your calculations in advance and it won't be possible to change them later.

   - The graph will be processed and optimized by the DL library before any computation.
   - Implemented in TensorFlow and other DL Toolkits

2. `Dynamic Graph`: you don't need to define your graph in advance (aka Computation Graph)
   - Execute operations that you want to use for data transformation on your actual data
   - During this the library will record the order of the operations performed and when you aks it to calculate gradients, it will unroll its history of operations, accumulating the gradients of the network parameters.
   - Implemented in PyTorch and some other

## Tensors and Gradients

PyTorch tensors have built-in gradient calculation and tracking machinery to simplify their use.

There are several attributes related to gradients that every tensor has:

`grad` - a property that holds a tensor of the same shape containing computed gradients

`is_leaf` - Equals `True` if this tensor was constructed by the user and `False` if the object is a result of function transformation (has a parent in the computation graph).

`requires_grad` - Equals `True` if this tensor requires gradients to be calculated.

    - This property is inherited from leaf tensors, which get this value from the tensor construction step

    - By default, the constructor has `requires_grad=False` so if you want gradients to be calculated you need to explicitly say so

```python
v1 = torch.tensor([1.0,1.0], requires_grad=True)
v2 = torch.tensor([2.0,2.0])
v_sum = v1 + v2
print(v_sum)

v_res = (v_sum*2).sum()
print(v_res)

# Check attributes of the tensors
print(v1.is_leaf, v2.is_leaf)

print(v_sum.is_leaf, v_res.is_leaf)

print(v1.requires_grad)

print(v2.requires_grad)

print(v_sum.requires_grad)

print(v_res.requires_grad)
```

> NOTE: For memory efficiency, gradients are stored only for leaf nodes with `requires_grad=True`. If you want to keep gradients in the non-leaf nodes, you need to call their `retain_grad()` method

```python
# Calculate the gradients
v_res.backward()
print(v1.grad)
```

# torch.nn Package

The `torch.nn` package provides access to predefined classes (derived from higher-level `nn.Module` base class) which provide sane default values and properly initialized weights

## Useful methods that `nn.Module` children provide:

- `parameters()` - returns an iterator of all variables that require gradient computation (i.e. model weights)
- `zero_grad()` - initializes all gradients of all parameters to zero
- `to(device)` - moves all module parameters to a given device (cpu or gpu)
- `state_dict()` - returns the dictionary with all module parameters and is useful for model serialization
- `load_state_dict()` - initializes the module with the state dictionary

### Example of `nn.Sequential`

```python
s = nn.Sequential(
    nn.Linear(2,5),
    nn.ReLU(),
    nn.Linear(5,20),
    nn.ReLU(),
    nn.Linear(20,10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1))

print(s)

out = s(torch.FloatTensor([[1,2]]))
print(out)
```

# Understanding the `nn.Module` base class

The `nn.Module` provides rich functionality to its child classes:

- Tracks all submodules that the current module includes

  - For example: your building block can have two feed-forward layers used somehow to perform the block's transformation. To keep track of (register) the submodule, you just need to assign it to the class's field

- Provides functions to deal with all parameters of the registered submodules

- Establishes the convention of `Module` application to data

- Provides additional advanced functionality - such as ability to register a hook function to tweak module transformation or gradients flow

These features allow us to nest submodules into higher-level models in a unified way.

## Creating a Custom Module

To create a custom module we usually only have to do two things:

1. Register submodules
2. Implement the `forward()` method

### Example of Creating Custom Module

```python
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


# Using custom module:

if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    print(net)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(out)
    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to("cuda"))
```
