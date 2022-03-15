import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_test(model, train_loader, test_loader, optimizer, loss_fn, device):
    
    """Perform one epoch of training and evaluate the model by doing one pass over a test dataset"""
    
    # train
    
    loss_train = 0
    
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):  
        
        #inputs = inputs.to(device)
        #target = target.to(device)
        
        output = model.forward(inputs.to(device).float())
        loss = loss_fn(output.float().view(output.shape[0]), target.to(device).float())
        
        with torch.no_grad():
            loss_train += loss.item()   
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    """"""
    
    # test
    
    loss_test = 0
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, target) in enumerate(test_loader):
            
            #inputs = inputs.to(device)
            #target = target.to(device)
            
            output = model.forward(inputs.to(device).float())
            loss = loss_fn(output.float().view(output.shape[0]), target.to(device).float())
            loss_test += loss.item()
            
            #print('batch: '+str(batch_idx) + ' , inputs shape:' + str(inputs.shape) + ' , loss = ' + str(loss.item()))
    
    return loss_train, loss_test