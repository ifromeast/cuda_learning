import torch
from triton_square import triton_square

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

compiled_square = torch.compile(torch.square)

a0 = torch.square(b)
a1 = square_2(b)
a2 = square_3(b)
a3 = compiled_square(b)
a4 = triton_square(b)