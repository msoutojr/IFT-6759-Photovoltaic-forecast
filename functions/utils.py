import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(model, train_loader, optimizer, loss_fn, device):
    
    """Perform one epoch of training the model by doing one pass over a train dataset"""
    
    loss_train = 0
    
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):  
        
        optimizer.zero_grad()
        
        inputs = inputs.to(device).float()
        target = target.to(device).float().view(target.shape[0],1)
        
        output = model.forward(inputs)
        
        loss = torch.sqrt(loss_fn(output, target))
        
        with torch.no_grad():
            loss_train += (loss.item()**2)*inputs.shape[0]   
        
        loss.backward()
        optimizer.step()
    
    return loss_train
 
    
def test(model, test_loader, optimizer, loss_fn, device):
    
    """Evaluate the model by doing one pass over a test dataset"""
    
    # test
    
    loss_test = 0
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, target) in enumerate(test_loader):
            
            inputs = inputs.to(device).float()
            target = target.to(device).float().view(target.shape[0],1)
            
            output = model.forward(inputs)
            
            loss = torch.sqrt(loss_fn(output, target))
            loss_test += (loss.item()**2)*inputs.shape[0]
            
    return loss_test
    
def test2(model, test_loader, optimizer, loss_fn, device):
    
    """Evaluate the model by doing one pass over a test dataset"""
    
    # test
    
    loss_test = 0
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, target) in enumerate(test_loader):
            
            inputs = inputs.to(device).float()
            target = target.to(device).float().view(target.shape[0],1)
            
            output = model.forward(inputs)
            
            loss = torch.sqrt(loss_fn(output, target))
            loss_test += (loss.item())
            
    return (loss_test)/(len(test_loader))
    

def train_gru(model, train_loader, optimizer, loss_fn, device, seq_len):
    
    """Perform one epoch of training the model by doing one pass over a train dataset"""

    loss_train = 0
    n=0
    model.train()
    
    for batch_idx, (inputs, target) in enumerate(train_loader):  
        
        optimizer.zero_grad()
        
        target = target.view(target.shape[0],1).float().to(device)
        
        if seq_len > 1:
            inputs = inputs.view(inputs.shape[0],seq_len,3,64,64).float().to(device)
        else:
            inputs = inputs.float().to(device)
            

        output = model(inputs).cuda()

        loss = loss_fn(output, target)
        
        with torch.no_grad():
            n_i = inputs.shape[0]
            loss_train += loss.item()*n_i 
            n += n_i        
        
        loss.backward()
        optimizer.step()
        
        
    return loss_train/n
 
    
def test_gru(model, test_loader, optimizer, loss_fn, device, seq_len):
    
    """Evaluate the model by doing one pass over a test dataset"""
    

    loss_test = 0
    
    n=0
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, target) in enumerate(test_loader):
            
            
            target = target.view(target.shape[0],1).float().to(device)
        
            if seq_len > 1:
                inputs = inputs.view(inputs.shape[0],seq_len,3,64,64).float().to(device)
            else:
                inputs = inputs.float().to(device)
            
            output = model(inputs)
            loss = torch.sqrt(loss_fn(output, target))
            
            n_i = inputs.shape[0]
            loss_test += (loss.item()**2)*n_i 
            n += n_i        

            
    return loss_test/n