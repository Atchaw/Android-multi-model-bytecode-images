import torch
import torch.nn as nn
import torch.optim as optim
from model import Net

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
model = Net()
# move tensors and criterion to GPU if CUDA is available
if train_on_gpu:
    model = model.cuda()
'''
def criterion():
    # specify loss function (categorical cross-entropy)
    criterionn =  nn.NLLLoss()
    if train_on_gpu:
        criterionn = criterionn.cuda()
    
    return criterionn
'''
def getOptimizer(learningRate, weightDecay):
    # specify optimizer
    return optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)