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
    
    return days_data

### LOAD AND PREPROCESS DATA ###
#this function creates a dictionary of dataloaders
def load_multiple_testing_data(n_batches=10,n_days_per_batch=1):
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
    
    test_list_size = len(XY_list_test)
          
    for batch in range(n_batches):
        dataset_test = TrainingUtils.FingerDatasetMultiDay(
            XY_list=[XY_list_test[batch]],
            predtype='pv',
            numfingers=2,
            numdelays=5,
            positioninput=False,
            last_timestep_recent=True,
            force_day_num=batch)##force day_idx to match batch number (this is how we ensure the correct input layer is used)

        # print(f'loaded {len(dataset_test)} test samples')

        # setup dataloaders
        num_test = len(dataset_test)
        loader_test = DataLoader(dataset_test, batch_size=int(num_test/2), sampler=sampler.SequentialSampler(range(num_test)))

        # Close the progress bar
        pbar.close()
        dataset_dict[batch] = {'batch_idx': batch, 'loader_test': loader_test}
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
            yhat = model.forward(x, day_idx=day_idx)
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
n_batches = 10
n_days_per_batch = 1 
dataset_dict = load_multiple_testing_data(n_batches,n_days_per_batch)
print(f'loaded {len(dataset_dict)} datasets')
# ----------------------------------------------------------------------------------------------------------------------
# force use of math_sdp to avoid errors
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# ----------------------------------------------------------------------------------------------------------------------

#directory where models are saved
directory_name = "trained_models/multiday_transformer_final_model_dicts"
    
# ----------------------------------------------------------------------------------------------------------------------
#init multiday models
fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)
one_day_file = os.path.join(fpath, '1days.pkl')
ten_day_file = os.path.join(fpath, '10days.pkl')
fifty_day_file = os.path.join(fpath, '50days.pkl')
hundred_day_file = os.path.join(fpath, '100days.pkl')
twohundred_day_file = os.path.join(fpath, '200days.pkl')

files = [ten_day_file, fifty_day_file, hundred_day_file, twohundred_day_file]
multiday_models = []

#load models
with open(one_day_file, 'rb') as f:
    data1 = pickle.load(f)
one_day_models = data1[0] #this is the list of models, so no .to(device)

for file in files:
    with open(file, 'rb') as f:
        temp_data = pickle.load(f)
    model = temp_data[0][(0,2,2)]['model'] 
    model.to(device)
    multiday_models.append(model)

model_num_per_day = [4]*10 + [3]*40 + [2]*50 +[1]*100

print('models loaded')
# ----------------------------------------------------------------------------------------------------------------------
results = np.zeros((5,4,n_batches))
for day in range(n_batches): 
    print(day)

    loader_test = dataset_dict[day]['loader_test']
    one_day_model = one_day_models[day]['model']
    one_day_model.to(device)

    #multiday_models= [ten_day_model, fifty_day_model, hundred_day_model, twohundred_day_model]
    num_models_today = model_num_per_day[day]
    
    for model,idx in zip(multiday_models[(4-num_models_today):],range(num_models_today)):
            test_corr = check_accuracy_multiday(model, loader_test)
            results[idx+(4-num_models_today),:,day] = test_corr[0]
    
    corr, _, _ = AnalysisTools.calc_model_performance(one_day_model, loader_test, normalize_y=False)
    results[4,:,day] = corr

    
fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory_name)
fname_out = 'Xday_by_day_results.pkl'
with open(os.path.join(fpath, fname_out), 'wb') as f:
    pickle.dump([results], f)
print(f'results saved to: {os.path.join(fpath, fname_out)}')
