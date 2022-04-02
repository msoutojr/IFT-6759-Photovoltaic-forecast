# Create Dataset and Dataloaders for sequence:

import os
import random
import numpy as np
from numpy import genfromtxt
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold

def date_dataframes(dates_train, dates_test):
        
        # datetime: year, month, day, hour, minute, second, microsecond, and tzinfo.

        date_train_list = np.zeros((dates_train.shape[0],5)).astype(int)
        date_test_list = np.zeros((dates_test.shape[0],5)).astype(int)

        for i in range(date_train_list.shape[0]):
              date_train_list[i] = np.array([dates_train[i].year, dates_train[i].month, dates_train[i].day, dates_train[i].hour, dates_train[i].minute])

        for i in range(date_test_list.shape[0]):
              date_test_list[i] = np.array([dates_test[i].year, dates_test[i].month, dates_test[i].day, dates_test[i].hour, dates_test[i].minute])

        date_train_list = pd.DataFrame(date_train_list)
        date_test_list = pd.DataFrame(date_test_list)

        return date_train_list, date_test_list

def create_maps(seq_len, dates_train, dates_test):

        date_train_list, dates_test = date_dataframes(dates_train, dates_test)

        # TRAIN MAP

        map_train = list(date_train_list.groupby([0,1,2]).agg({3:'count', 1:'count'})[3])

        prov1 = map_train[0]
        map_train[0]=0

        for i in range(1,len(map_train)):
          prov2 = map_train[i]
          map_train[i] = prov1
          prov1 += prov2 
        
        start_day = map_train.copy()
        start_day.append(prov1) # add the final index to recover the last range day
        
        new_list = []
        for i in range(seq_len):
          new_list.append(x+i for x in map_train)

        map_train = map_train + new_list

        # TEST MAP
        
        map_test = list(dates_test.groupby([0,1,2]).agg({3:'count', 1:'count'})[3])

        prov1 = map_test[0]
        map_test[0]=0

        for i in range(1,len(map_test)):
          prov2 = map_test[i]
          map_test[i] = prov1
          prov1 += prov2 

        new_list = []
        for i in range(seq_len):
          new_list.append(x+i for x in map_test)

        map_test = map_test + new_list


        return map_train, map_test, start_day

class ImageData(Dataset):
    
    # dataset class

    def __init__(self, inputs, labels, map_start_day, seq_len):
        'Initialization'
        self.inputs = inputs
        self.labels = labels
        self.map_start_day = map_start_day
        self.seq_len = seq_len

    def __len__(self):
        'Denotes the total number of samples'
        return self.inputs.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        
        X = self.inputs[index]
        y = self.labels[index]#.view(1,1)
        
        if self.seq_len > 1:
            if index in self.map_start_day:
                X = self.inputs[index].repeat(self.seq_len,1,1)
            else:  
                X = self.inputs[index]
                for i in range(self.seq_len-1):
                                X = torch.cat( (self.inputs[index-i],X),dim=0)
        return X, y





def create_dataloaders(seq_len, dates_train, dates_test, set_train, set_test, target_train, target_test, 
                       batch_size = 128 , batch_size_eval = 256, split = True):
            
            # Create Dataloaders
            
            
            #create maps to detect first n images of the day (n = sequence length) 
            map_train, map_test, start_day = create_maps(seq_len, dates_train, dates_test)
            
            # Create Datasets
            train_data_base = ImageData(torch.tensor(set_train), torch.tensor(target_train), map_train, seq_len)
            test_data_base = ImageData(torch.tensor(set_test), torch.tensor(target_test), map_test, seq_len)
            
            if split == True:
                    
                        days = len(start_day)-1 # there is an extra value at the end
                        
                        # Shuffle days and do k-fold without shuffle. There will be some days instersections
                        
                        shuf_days = random.sample(range(days), days)
                        
                        shuf_days_index = []
                        
                        for i in shuf_days:
                          shuf_days_index = shuf_days_index + list(range(start_day[i],start_day[i+1]))
                        
                        # split 10-fold
                        n_splits = 10
                        kf = KFold(n_splits=n_splits, shuffle=False)
                        
                        train_dataset = []
                        valid_dataset = []
                        
                        for train_index, valid_index in kf.split(shuf_days_index):  
                            train_dataset.append(Subset(train_data_base, train_index))
                            valid_dataset.append(Subset(train_data_base, valid_index))  
                        
                        test_dataset = test_data_base  
                        
                        train_loaders = []
                        valid_loaders = []
                        
                        #create dataloaders
                        
                        for i in range(len(train_dataset)):
                        
                              train_loaders.append(
                              DataLoader(
                                  train_dataset[i],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2
                              ))
                              
                              valid_loaders.append(
                              DataLoader(
                                  valid_dataset[i],
                                  batch_size=batch_size_eval,
                                  shuffle=False,
                                  num_workers=2
                              ))
                        
                        test_loader = DataLoader(
                          test_dataset,
                          batch_size=batch_size_eval,
                          shuffle=False,
                          num_workers=2
                        )
            else:
                        
                        train_loaders = []
                        valid_loaders = []
                        
                        train_loaders.append(DataLoader(
                                  train_data_base,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2
                                ))
                      
                        test_loader = DataLoader(
                                  test_data_base,
                                  batch_size=batch_size_eval,
                                  shuffle=False,
                                  num_workers=2
                                )
                        
                        valid_loaders.append(test_loader)
                      
                      
            return train_loaders, valid_loaders, test_loader
