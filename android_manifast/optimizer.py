import torch.nn as nn
import torch.optim as optim

def getCriterion():
    # specify loss function (categorical cross-entropy)
    criterion = nn.NLLLoss()

def getOptimizer(learningRate, weightDecay ):
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)