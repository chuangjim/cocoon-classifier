import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time
import pickle

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(128, 2))
print(model)
# for name,param in model.named_parameters():
#     if param.requires_grad == True:
#         print("\t",name)