import torch
from model.cnn import simplecnn

x = torch.randn(32, 3, 224, 224)
model = simplecnn(num_class=4)
output = model(x)
print(output.shape)