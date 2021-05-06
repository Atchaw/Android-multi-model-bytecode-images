import torch
import torch.nn as nn
import torch.optim as optim

def criterion():
    # specify loss function (categorical cross-entropy)
    return nn.NLLLoss()

def optimizer(learningRate, weightDecay ):
    # specify optimizer
    return optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)