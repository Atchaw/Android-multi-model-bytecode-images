import torch
import torch.nn as nn

# defining the model architecture
class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        
        # Conv-Conv-Pool-Conv-Conv-Pool
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer 
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining a 2D convolution layer 
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            # Defining a 2D convolution layer 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Defining a 2D convolution layer 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #Defining a Dropout
            nn.Dropout(0.5),
        )

        self.linear_layers = nn.Sequential(
            # Defining a Linear layer 
            nn.Linear(8*8*64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            #Defining a Dropout
            nn.Dropout(0.5),

            # Defining a Linear layer 
            nn.Linear(64, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(inplace=True),

            #Defining a Dropout
            nn.Dropout(0.25),

            # Defining another Linear layer 
            nn.Linear(40, 2),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x