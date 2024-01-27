import torch
import timeit
import random

# x = torch.ones(50000000,device='cuda')
x = torch.ones(50000000,device='mps')
print(timeit.timeit(lambda:x*random.randint(0,100),number=1))