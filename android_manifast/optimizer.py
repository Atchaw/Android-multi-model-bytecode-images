import torch.nn as nn
import torch.optim as optim

def getOptimizer(learningRate, weightDecay ):
    # specify loss function (categorical cross-entropy)
    criterion = nn.NLLLoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)