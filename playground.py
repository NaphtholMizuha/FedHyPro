from utils.param import quantize, unquantize
import torch

tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
dic = {'a': tensor}
print(dic)
tq = quantize(dic)
print(tq)
print(unquantize(tq))
