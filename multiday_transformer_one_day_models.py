### IMPORTS ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import yaml
from datetime import datetime
from tqdm import tqdm

# import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from pybmi.utils import AnalysisTools
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torch.optim as optim
from pybmi.utils import TrainingUtils
from pybmi.utils.TrainingUtils import get_cuda_device, get_server_data_path
import pybmi
from pybmi.decoders import TFMDecoders, RNNDecoders

dtype = torch.float
device = get_cuda_device()
print(str(torch.cuda.device_count())+" GPUs in use")

# ----------------------------------------------------------------------------------------------------------------------
### FUNCTION TO EXTRACT SBP and KINEMATICS ###

#train split default is 0.8
#equally split val and test of remaining data
def extract_sbp_kinematics(data,train_split=0.8):
    # Check if data is a tuple and extract the dictionary if necessary
    if isinstance(data, tuple):
        if data[0] is None and isinstance(data[1], dict):
            data = data[1]
        elif isinstance(data[0], dict):
            data = data[0]
        else:
            print(f"Unexpected data structure: {type(data)}")
            return None

    # Extract relevant data
    val_split = 0.5 - train_split/2
    finger_kinematics = torch.as_tensor(data['finger_kinematics'])
    sbp = torch.as_tensor(data['sbp'])
    num_train = int(sbp.shape[0] * train_split)
    x_train = sbp[:num_train, :]
    y_train = finger_kinematics[:num_train, :]
    num_val = int(sbp.shape[0] * val_split)
    x_val = sbp[num_train:num_train+num_val, :]
    y_val = finger_kinematics[num_train:num_train+num_val, :]
    x_test = sbp[num_train+num_val:, :]
    y_test = finger_kinematics[num_train+num_val:, :]####Bug
    
    # normalize X
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train = (x_train - x_mean) / (x_std + 1e-6)
    x_val = (x_val - x_mean) / (x_std + 1e-6)
    x_test = (x_test - x_mean) / (x_std + 1e-6)
    # normalize Y (optional)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    y_train = (y_train - y_mean) / (y_std + 1e-6)
    y_val = (y_val - y_mean) / (y_std + 1e-6)
    y_test = (y_test - y_mean) / (y_std + 1e-6)
    
    days_data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,
    }
    # for k, _ in data.items():
    #     print(k)
    
    return days_data

### LOAD AND PREPROCESS DATA ###
def load_multiple_big_training_data(n_batches=10,n_days_per_batch=1):
    # Approximate total number of datasets
    total_datasets = 415
    dataset_dict = {}
    # Create a progress bar
    pbar = tqdm(total=total_datasets, desc="Processing datasets")

    # Path to the folder containing pkl files
    data_folder = './preprocessing_092024'

    # Get list of pkl files
    pkl_files = sorted(glob.glob(os.path.join(data_folder, '*.pkl')))

    # init lists to hold data (filled in below)
    XY_list_train = []
    XY_list_val = []
    XY_list_test = []

    # Process each pkl file
    num_days_to_include = n_days_per_batch*n_batches
    counter = 0

    for file in pkl_files:
        # Extract date from filename (assuming format like 'YYYY-MM-DD_data.pkl')
        date = pd.to_datetime(os.path.basename(file).split('_')[0])

        # Load data
        with open(file, 'rb') as f:
            data = pickle.load(f)
        # Compute channel tuning
        try:
            day_data = extract_sbp_kinematics(data)
            if day_data is not None:
                # save data to list
                XY_list_train.append((day_data['x_train'], day_data['y_train']))
                XY_list_val.append((day_data['x_val'], day_data['y_val']))
                XY_list_test.append((day_data['x_test'], day_data['y_test']))
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

        # Update the progress bar
        pbar.update(1)
        
        #include only the first num_days_to_include days
        counter += 1
        if counter == num_days_to_include:
            break
        
    # setup datasets (which add time history)?
    batch_size = 64
    
    train_list_size = len(XY_list_train)
    val_list_size = len(XY_list_val)
    test_list_size = len(XY_list_test)
    train_idx = int(train_list_size/n_batches)
    val_idx = int(val_list_size/n_batches)
    test_idx = int(test_list_size/n_batches)
    for batch in range(n_batches):
        dataset_train = TrainingUtils.FingerDatasetMultiDay(
            XY_list=XY_list_train[batch*train_idx:(batch+1)*train_idx],
            predtype='pv',
            numfingers=2,
            numdelays=5,# 5 or so for transformer
            positioninput=False,
            last_timestep_recent=True)
        dataset_val = TrainingUtils.FingerDatasetMultiDay(
            XY_list=XY_list_val[batch*val_idx:(batch+1)*val_idx],
            predtype='pv',
            numfingers=2,
            numdelays=5,# 5 or so for transformer
            positioninput=False,
            last_timestep_recent=True)

        # print(f'loaded {len(dataset_train)} training samples')
        # print(f'loaded {len(dataset_val)} validation samples')
        # print(f'loaded {len(dataset_test)} test samples')

        # setup dataloaders
        num_train = len(dataset_train)
        num_val = len(dataset_val)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler.RandomSampler(range(num_train)))
        loader_val = DataLoader(dataset_val, batch_size=int(num_val/2), sampler=sampler.SequentialSampler(range(num_val)))
        
        # Close the progress bar
        pbar.close()
        dataset_dict[batch] = {'batch_idx': batch, 'loader_train': loader_train, 'loader_val': loader_val}
    return dataset_dict

def run_model_forward_multiday(model, loader):
    with torch.no_grad():
        # get batch data (we assume there's only 1 batch) TODO: concat multiple batches
        for batch in loader:
            
            x = batch['chans']
            y = batch['states']
            day_idx = batch['day_idx']
            
            model.eval()
            x = x.to(device) ################ successful example of temporalily moving tensor for cpu to gpu and back
            yhat = model.forward(x, day_idx)
            x = x.to('cpu')
    
            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]
                
            return y, yhat, day_idx
        

def check_accuracy_multiday(model, loader):
    y, yhat, day_idx = run_model_forward_multiday(model, loader)

    y = y.cpu().detach().numpy()
    yhat = yhat.cpu().detach().numpy()
    day_idx = day_idx.detach().numpy()
    corrs = []
    
    for i in np.unique(day_idx):
        this_day = (day_idx==i).squeeze()
        thiscorr = AnalysisTools.pairedcorrcoef(y[this_day, :], yhat[this_day, :])
        corrs.append(thiscorr)
    
    return corrs

# ----------------------------------------------------------------------------------------------------------------------
#load training data
n_batches = 200
n_days_per_batch = 1 
dataset_dict = load_multiple_big_training_data(n_batches,n_days_per_batch)
# ----------------------------------------------------------------------------------------------------------------------
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

#init model
MODEL_TYPE = 'TFM'                  
config_name = 'TFM_xnorm_ynorm'

loss_func = nn.MSELoss()

# load config
config_fpath = os.path.join(os.path.dirname(pybmi.__file__), 'decoders/configs.yaml')
with open(config_fpath) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)[config_name]
    training_params = config['training_params']


# setup general params
input_size = 96
num_states = 4
verbose = training_params['verbose']
# ----------------------------------------------------------------------------------------------------------------------

#create a directory to store results
directory_name = "multiday_transformer_200x1_day_models_100k_maxiter"

# Create the directory
try:
    os.mkdir(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
except FileExistsError:
    print(f"Directory '{directory_name}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{directory_name}'.")
except Exception as e:
    print(f"An error occurred: {e}")
    
# ----------------------------------------------------------------------------------------------------------------------
results ={} #store results

for batch_idx in range(n_batches): #lr is decided

    lr = float(training_params['learning_rate'])
    nhid = 1000 #config['hidden_size']
    nlayers = 4 #config['num_layers']

    loader_train = dataset_dict[batch_idx]['loader_train']
    loader_val = dataset_dict[batch_idx]['loader_val']

    #build model
    tfm_model = TFMDecoders.TransformerModel(input_size, num_states, enc_nhead=config['num_heads'],
                                            enc_nhid=nhid,
                                            enc_nlayers=nlayers,
                                            dropout=config['dropout_p']).to(device)

    #build optimizer
    optimizer = optim.Adam(tfm_model.parameters(),
                        lr=lr,
                        weight_decay=float(training_params['weight_decay']))
    if training_params['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                        patience=training_params['scheduler_patience'])
        
    # Train
    final_iter, val_loss, corr, history = TrainingUtils.train_model(
            tfm_model,
            optimizer,
            loss_func,
            loader_train,
            loader_val,
            check_accuracy_iters=200,
            verbose=True,
            scheduler=None, # None or scheduler
            min_lr=lr/2,
            max_iter=100000,#10000
            plot_progress=False,
            plot_simple=True,
            return_history=True,
            multiday=False  #single day training
    )
    
    tfm_model.to(device='cpu')
    results[batch_idx] = {'batch': batch_idx, 'lr': lr, 'enc_nhid': nhid, 'enc_nlayers': nlayers, 'final_iter': final_iter, 'val_loss': val_loss, 'history': history, 'model': tfm_model}
    
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)
    fname_out = str(batch_idx)+'.pkl'
    with open(os.path.join(fpath, fname_out), 'wb') as f:
        pickle.dump([results], f)
    print(f'results saved to: {os.path.join(fpath, fname_out)}')
