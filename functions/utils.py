import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(model, train_loader, optimizer, loss_fn):
    
    """Perform one epoch of training."""
    
    loss_train = 0
    n = 0
    
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):  
        
        output = model.forward(inputs.float())
        loss = loss_fn(output.float(), target.float())
        
        with torch.no_grad():
            loss_train += loss   
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n += 1
    return loss_train/n


def test(model, test_loader, loss_fn):
    
    """Evaluate the model by doing one pass over a dataset"""
    
    loss_test = 0
    n = 0
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, target) in enumerate(test_loader):

            output = model.forward(inputs.float())
            loss = loss_fn(output.float(), target.float())
            loss_test += loss
            n += 1
    
    return loss_test/n