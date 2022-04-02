
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
        x = x.cuda().float()
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



class SUNSET_GRU4(nn.Module):
    
    """

    """
    def __init__(self, conv_layers=2, convGRU_layers = 3, fc_layers = 2 ):
        
        super(SUNSET_GRU4, self).__init__()
        
        import models.convGRU as convGRU

        if conv_layers > 2:
            conv_layers = 2
            print('max conv layers = 2')
        self.conv_layers = conv_layers
        
        if convGRU_layers > 3:
            convGRU_layers = 3
            print('max number of convGRU layers = 3')
        self.convGRU_layers = convGRU_layers
        
        if fc_layers > 2:
            fc_layers == 2
            print('max fc layers = 2')
        self.fc_layers = fc_layers
    
        if conv_layers == 2:
            chanels = [3,24,48]
        elif conv_layers == 1:
            chanels = [3,24]
        else:
            chanels = [3]
        
        #Conv Layers            
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size = 3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size = 3, padding="same")
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(48)
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        
        # ConvGRU        
        input_size = ( int(64/(2**conv_layers)), int(64/(2**conv_layers)) )
        input_dim = int(chanels[-1])
        hidden_dim_set = [32,32,32]
        kernel_size_set = [(3,3),(3,3),(3,3)]
        hidden_dim = []
        kernel_size = []
        for i in range(convGRU_layers):
            hidden_dim.append(int(hidden_dim_set[i]))
            kernel_size.append(kernel_size_set[i])
        num_layers = convGRU_layers
        dtype = torch.cuda.FloatTensor
        batch_first=True
        bias = False
        return_all_layers=False
        self.GRU = convGRU.ConvGRU(input_size,input_dim,hidden_dim,kernel_size,num_layers,dtype,batch_first,bias,return_all_layers).cuda()
        
        
        # FC layers
        if convGRU_layers > 0:
            dim_0 = int(hidden_dim[-1]*input_size[0]*input_size[1])
        else:
            dim_0 = int(chanels[-1]*input_size[0]*input_size[1])
        if fc_layers == 0:
            dim_reg = dim_0
        else:
            dim_reg = 1024
        self.drop_rate = 0.4
        self.fc1 = nn.Linear(dim_0, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop1 = nn.Dropout(self.drop_rate)
        self.drop2 = nn.Dropout(self.drop_rate)
        self.reg = nn.Linear(dim_reg, 1)

    
    
    def forward(self, inputs=None, device="cuda:0", hidden_state=None):
        
        batch = inputs.shape[0]
        if inputs.dim() == 5:
            seq_len = inputs.shape[1]
        else:
            seq_len = 1
        
        C = inputs.shape[-3]
        H = inputs.shape[-2]
        W = inputs.shape[-1]
        
        if self.conv_layers > 0:
            x = inputs.view(batch*seq_len,C,H,W)
        else:
            x = inputs
        
        if self.conv_layers > 0:
                x = self.conv1(x)
                x = F.relu(x)
                x = self.bn1(x)
                x = self.maxpool1(x)
        
        if self.conv_layers == 2:
                x = self.conv2(x)
                x = F.relu(x)
                x = self.bn2(x)
                x = self.maxpool2(x)
        
        
        C = x.shape[-3]
        H = x.shape[-2]
        W = x.shape[-1]
        
        if self.convGRU_layers > 0:
                x = x.view(batch,seq_len,C,H,W)
                x,h = self.GRU(x)
                x = torch.flatten(x[0][:,-1,:,:,:], start_dim=1).to(device)
        else:
                x = torch.flatten(x, start_dim=1).to(device)
        
        
        if self.fc_layers > 0:
                x = self.fc1(x)        
                x = F.relu(x)
                x = self.drop1(x)
        
        if self.fc_layers == 2:
                x = self.fc2(x)
                x = F.relu(x)
                x = self.drop2(x)
        
        output = self.reg(x)
        return output
        
        
        