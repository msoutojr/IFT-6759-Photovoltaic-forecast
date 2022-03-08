
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SUNSETBase(nn.Module):
    
    """

    """

    def __init__(self):
        super(SUNSETBase, self).__init__()
        
        self.drop_rate = 0.4
        
        self.drop1 = nn.Dropout(self.drop_rate)
        self.drop2 = nn.Dropout(self.drop_rate)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size = 3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size = 3, padding="same")

        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(12288, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        self.reg = nn.Linear(1024, 1)

    
    def forward(self, x):
        
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!
        """
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
                
        x = x.view(x.size(0), -1) 

        x = self.fc1(x)        
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        
        x = self.reg(x)
        
        return x

        
class SUNSET_Sunny(nn.Module):
    
    """

    """

    def __init__(self):
        super(SUNSET_Sunny, self).__init__()
        
        self.drop_rate = 0.4
        
        self.drop1 = nn.Dropout(self.drop_rate)
        self.drop2 = nn.Dropout(self.drop_rate)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size = 3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size = 3, padding="same")

        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

        self.reg = nn.Linear(12288, 1)

    
    def forward(self, x):
        
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!
        """
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
                
        x = x.view(x.size(0), -1) 

        x = self.reg(x)
        
        return x

   
class SUNSET_Cloudy(nn.Module):
    
    """

    """

    def __init__(self):
        super(SUNSET_Cloudy, self).__init__()
        
        self.drop_rate = 0.4
        
        self.drop1 = nn.Dropout(self.drop_rate)
        self.drop2 = nn.Dropout(self.drop_rate)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size = 3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size = 3, padding="same")
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=72, kernel_size = 3, padding="same")
        self.conv4 = nn.Conv2d(in_channels=72, out_channels=96, kernel_size = 3, padding="same")

        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn3 = nn.BatchNorm2d(72)
        self.bn4 = nn.BatchNorm2d(96)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(2)
        
        #self.fc1 = nn.Linear(12288, 1024)
        #self.fc2 = nn.Linear(1024, 1024)
        
        self.reg = nn.Linear(1536, 1)

    
    def forward(self, x):
        
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!
        """
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
       
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.maxpool4(x)
        
        x = x.view(x.size(0), -1) 

        x = self.reg(x)
        
        return x


class SUNSET_Overcast(nn.Module):
    
    """

    """

    def __init__(self):
        super(SUNSET_Overcast, self).__init__()
        
        self.drop_rate = 0.4
        
        self.drop1 = nn.Dropout(self.drop_rate)
        self.drop2 = nn.Dropout(self.drop_rate)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size = 3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size = 3, padding="same")

        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(12288, 1024)
        
        self.reg = nn.Linear(1024, 1)

    
    def forward(self, x):
        
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!
        """
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
                
        x = x.view(x.size(0), -1) 

        x = self.fc1(x)        
        x = F.relu(x)
        x = self.drop1(x)
        
        
        x = self.reg(x)
        
        return x