import os
import random
import numpy as np
from numpy import genfromtxt
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import sys
import shutil
import warnings
from datetime import datetime

import models.SUNSET as SUNSET
import models.convGRU as convGRU
import functions.utils as utils
import functions.load_creator as load_creator
import models.convLSTM as convLSTM

# Create device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB", flush=True)

# set Random seed
seed = 10
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# define folder location
dir_path = os.getcwd()
data_folder = os.path.join(dir_path, "data")
model_folder = os.path.join(dir_path, "models")

# define file location
images_trainval_path = os.path.join(data_folder,'images_trainval.npy')
pv_log_trainval_path = os.path.join(data_folder,'pv_log_trainval.npy')
datetime_trainval_path = os.path.join(data_folder,'datetime_trainval.npy')

images_test_path = os.path.join(data_folder,'images_test.npy')
pv_log_test_path = os.path.join(data_folder,'pv_log_test.npy')
datetime_test_path = os.path.join(data_folder,'datetime_test.npy')

# load PV output and images for the trainval set
pv_log_trainval = np.load(pv_log_trainval_path)
images_trainval = np.load(images_trainval_path)
images_trainval = np.transpose(images_trainval, (0,3,1,2))
datetimes_trainval = np.load(datetime_trainval_path, allow_pickle=True)

# load PV output and images for the test set
pv_log_test = np.load(pv_log_test_path)
images_test = np.load(images_test_path)
images_test = np.transpose(images_test, (0,3,1,2))
datetimes_test = np.load(datetime_test_path, allow_pickle=True)# get weather classification

# train
w_class_train_path = os.path.join(data_folder,'train_label_3_classes_d.csv')
w_class_train = pd.read_csv(w_class_train_path)

index_train_sunny = np.where(w_class_train == 'Sunny')[0].tolist()
index_train_cloudy = np.where(w_class_train == 'Cloudy')[0].tolist()
index_train_overcast = np.where(w_class_train == 'Overcast')[0].tolist() 

# test
w_class_test_path = os.path.join(data_folder,'test_label_3_classes_d.csv')
w_class_test = pd.read_csv(w_class_test_path)

index_test_sunny = np.where(w_class_test == 'Sunny')[0].tolist()
index_test_cloudy = np.where(w_class_test == 'Cloudy')[0].tolist()
index_test_overcast = np.where(w_class_test == 'Overcast')[0].tolist()

# Sunny
pv_log_trainval_sunny = pv_log_trainval[index_train_sunny] 
images_trainval_sunny = images_trainval[index_train_sunny]
datetimes_trainval_sunny = datetimes_trainval[index_train_sunny]

pv_log_test_sunny = pv_log_test[index_test_sunny]
images_test_sunny = images_test[index_test_sunny]
datetimes_test_sunny = datetimes_test[index_test_sunny]

# Cloudy
pv_log_trainval_cloudy = pv_log_trainval[index_train_cloudy] 
images_trainval_cloudy = images_trainval[index_train_cloudy]
datetimes_trainval_cloudy = datetimes_trainval[index_train_cloudy]

pv_log_test_cloudy = pv_log_test[index_test_cloudy]
images_test_cloudy = images_test[index_test_cloudy]
datetimes_test_cloudy = datetimes_test[index_test_cloudy]

# Overcast

pv_log_trainval_overcast = pv_log_trainval[index_train_overcast] 
images_trainval_overcast = images_trainval[index_train_overcast]
datetimes_trainval_overcast = datetimes_trainval[index_train_overcast]

pv_log_test_overcast = pv_log_test[index_test_overcast]
images_test_overcast = images_test[index_test_overcast]
datetimes_test_overcast = datetimes_test[index_test_overcast]

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


# LSTM Training (Automated)

# Building training configurations
lstm_configs = []
for class_name in ["Sunny", "Cloudy", "Overcast"]:
    for convLSTM_layers in range(1,4):
        for fc_layers in range(3):
            a, b = convLSTM_layers, fc_layers
            # exclude the ones we've already done
            num_epochs = 100 if class_name == 'Overcast' else 30
            # class_name, seq_len=2, convLSTM_layers, fc_layers, num_epochs, learning_rate=1e-5, weight_decay=1e-5, conv_layers=0
            config = [class_name, 2, convLSTM_layers, fc_layers, num_epochs, 1e-5, 1e-5, 0]
            lstm_configs.append(config)

# Generic code for training an LSTM model using the given configuration
def training(config):
    class_name, seq_len, convLSTM_layers, fc_layers, num_epochs, learning_rate, weight_decay, conv_layers = config

    learning_rate = 1e-5
    batch_size = 64
    batch_size_eval = 64

    model = SUNSET.SUNSET_LSTM(conv_layers=conv_layers, convLSTM_layers=convLSTM_layers, fc_layers=fc_layers).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters() , lr=learning_rate, weight_decay=weight_decay)

    if class_name == 'Sunny':
        # datetimes_trainval, datetimes_test, images_trainval, images_test, pv_log_trainval, pv_log_test
        dt_tv, dt_te, im_tv, im_te, pv_tv, pv_te = datetimes_trainval_sunny, datetimes_test_sunny, images_trainval_sunny, images_test_sunny, pv_log_trainval_sunny, pv_log_test_sunny
    elif class_name == "Cloudy":
        dt_tv, dt_te, im_tv, im_te, pv_tv, pv_te = datetimes_trainval_cloudy, datetimes_test_cloudy, images_trainval_cloudy, images_test_cloudy, pv_log_trainval_cloudy, pv_log_test_cloudy
    else:
        dt_tv, dt_te, im_tv, im_te, pv_tv, pv_te = datetimes_trainval_overcast, datetimes_test_overcast, images_trainval_overcast, images_test_overcast, pv_log_trainval_overcast, pv_log_test_overcast
                                                             
                                                             
    train_loaders, valid_loaders, test_loader = load_creator.create_dataloaders(seq_len,
                                                            dt_tv, dt_te, im_tv, im_te, pv_tv, pv_te, batch_size=batch_size, batch_size_eval=batch_size_eval, split=False)
           
    for split in range(len(train_loaders)):  
    
        model.apply(weight_reset)

        Train_losses = []
        Valid_losses = []
        Test_losses = []

        for epoch in range(num_epochs):
            print(f"\t{datetime.now()}: Started Epoch {epoch+1}.", flush=True)
            l_train   =  utils.train_lstm(model, train_loaders[split], optimizer, loss_fn, device, seq_len)
            l_val     =  utils.test_lstm(model, valid_loaders[split], optimizer, loss_fn, device, seq_len)     
            l_test    =  utils.test_lstm(model, test_loader, optimizer, loss_fn, device, seq_len)

            Train_losses.append(math.sqrt(l_train))
            Valid_losses.append(math.sqrt(l_val))
            Test_losses.append(math.sqrt(l_test))

            # Early Stopping:
            if epoch > 10 and (epoch - Valid_losses.index(min(Valid_losses)) > 3 ):
                print(f"Early stopped after Epoch {epoch+1}!", flush=True)
                break

    return min(Test_losses)

def clear_memory():
    """
    Empties the PyTorch cache and prints out the currently available VRAM.
    """
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(f"Memory available: {(r-a)/1e+9:.2f}GB ({r/1e+9:.2f}/{a/1e+9:.2f})")
    torch.cuda.empty_cache()

# Run each config separately
import sys
i = int(sys.argv[1])
config = lstm_configs[i]
try:
    class_name, convLSTM_layers, fc_layers  = config[0], config[2], config[3]
    print(f"Started: {class_name}-{convLSTM_layers}-{fc_layers} (config{i})", flush=True)
    result = training(config)
    print(f"{class_name}-{convLSTM_layers}-{fc_layers}: {result}", flush=True)
except RuntimeError:
    print(f"Skipped config {i} due to insufficient video memory!", flush=True)
clear_memory()