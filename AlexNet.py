import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__ (self, num_classes = 1000):
        super(AlexNet,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,
                      kernel_size=11, stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384,384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.MaxPool2d(kernel_size=3, stride=2),      



        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256*6*6), out_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=num_classes),
        )

        self.init_bias()

    def init_bias(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        
    
    def foward(self, x):
        
        x = self.features(x)
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x
    
model = AlexNet(num_classes=10)  # For example, if you have 10 classes instead of 1000
print(model)