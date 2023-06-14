"""
Representation Learner Dataset
"""
import pdb
import collections, copy, os, random, sys, torch, numpy as np, pandas as pd, pickle
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
from typing import Callable, Dict, Sequence, ClassVar, Set

import json 

from ..constants import *
from ..utils import *

import time
import random

import warnings

class PatientDataset(Dataset):
    """An example dataset class"""
    def __init__(
        self,
        min_seq_len:   int = 8,
        max_seq_len:   int = 8,
        eval_seq_len:   int = 8,
        seed:          int = 0,
        task:          str = 'ssl', # specify SSL for self sup learning, or other tasks. The universe of tasks is defined in ../constants.py
        signal_seconds:int = 10,
        signal_mask:   float = 0.25,
        history_cutout_prob: float = 0.8,
        history_cutout_frac: float = 0.5,
        spatial_dropout_rate: float= 0.1,
        corrupt_rate: float= 0.6,
    ):
        """
        dataset maker is in
        """
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.eval_seq_len = eval_seq_len
        self.seed = seed
        self.task = task
        self.signal_seconds = signal_seconds
        self.signal_mask = signal_mask
        self.history_cutout_prob = history_cutout_prob
        self.history_cutout_frac = history_cutout_frac
        self.spatial_dropout_rate = spatial_dropout_rate
        self.corrupt_rate = corrupt_rate


    def __len__(self):
        # Just a placeholder -- will need to be properly defined based on some list of filenames, for example
        return 1024

    def subsample(self, frac, seed=0):
        # NOTE this functionality can be implemented for a specific dataset if needed.
        return

    def make_signals_traj(self, data, start_time, end_time, index=0):
        ret = []

        times = list(data.keys())

        for t in range(start_time, end_time):

            t_mapped = times[t]
            
            # This is written assuming we just have a single lead: leadII
            ecg = data[t_mapped]['leadII']

            ecg_len = len(ecg)

            if self.task == 'ssl':
                # At SSL time, we take disjoint chunks to form the views 
                # NOTE: Assume here that we have 30s of data stored in ecg sampled at 125Hz
                if index==0:
                    start = 0
                    end = 1+ecg_len//2
                else:
                    start = 1+ecg_len//2
                    end = 1+ecg_len
                start_sample = np.random.randint(start, end - 125*self.signal_seconds)
                end_sample = start_sample + 125*self.signal_seconds
            else:
                # At finetuning time, take the first signal_seconds worth.
                start_sample = 0
                end_sample = start_sample + 125*self.signal_seconds

            # NOTE: here, we could do preprocessing of the signal -- clipping, bandpass filtering, etc.
            # Leaving out for now.

            ecg = ecg[start_sample:end_sample]
            # Insert a channel dimension. Right now, ECG is a vector of size NumSamples
            ecg = np.expand_dims(ecg, 0)
            ret.append(ecg)

        ret = np.stack(ret, axis=0)
        return ret


    def make_structured_data_traj(self, vitals_data, start_time, end_time):
        # NOTE this is simplified, assuming that everything is measured, no missingness, etc
        # In reality, we might need to do some kind of imputation etc. 

        times = list(vitals_data.keys())

        ret_struct = []
        for t in range(start_time, end_time):
            t_mapped = times[t]
            ret_struct.append(vitals_data[t_mapped])
        
        return np.stack(ret_struct, axis=0)

       
    def make_static_data(self, struc_data):
        # NOTE Like with the structured data traj, this is super simplified. 
        # Might want additional preprocessing, for example!
        return struc_data
        

    def mask_augmentation(self, signal):
        # Signal is of shape SEQ_Len x Num_CH x num_samples
        # For each seq len and ch, generate up to self.signal_mask and apply
        crop_rate = self.signal_mask
        T, C, S = signal.shape
        if crop_rate == 0: 
            return signal, np.zeros(T)
        for t in range(T):
            for c in range(C):
                crop_len = int(crop_rate*S)
                crop_start = np.random.randint(0, S)
                stdval = 0.5
                noise = 0.5*stdval*np.random.randn(crop_len)
                if crop_start + crop_len <= S:
                    signal[t, c, crop_start:crop_start+crop_len] = noise
                else:
                    remainder = crop_len - (S-crop_start)
                    signal[t, c, crop_start:S] = noise[:S-crop_start]
                    signal[t, c, 0:remainder] = noise[S-crop_start:]

        return signal, None
    
    def struct_timeseries_augment(self, data):
        # Signal is of shape SEQ_Len x Num_feat
        # Tabular time series augmentation applies cutout augment then channel dropout
        
        crop_prob = self.history_cutout_prob
        crop_frac = self.history_cutout_frac

        # NOTE these are hardcoded for now, since we have random data.
        vitals_means = 0
        vitals_stds = 1

        T, C = data.shape
        
        masked = np.zeros_like(data)

        if crop_prob != 0:
            for c in range(C):
                if np.random.uniform() < crop_prob:
                    crop_len = int(crop_frac*T)
                    crop_start = np.random.randint(0, T - crop_len + 1)

                    data[crop_start:crop_start+crop_len, c] = np.nan
                    masked[crop_start:crop_start+crop_len,c] = 1
            # Now do forward filling and reverse filling (in case we crop out seq start)
            data = pd.DataFrame(data).fillna(method='ffill').values
            data = pd.DataFrame(data).fillna(method='bfill').values
        
        # Add noise
        for c in range(C):
            data[:,c] += np.random.normal(size=T)*vitals_stds*0.1
        
        # Do channel dropout -- impute removed values with the mean
        for c in range(C):
            if np.random.uniform() < self.spatial_dropout_rate:
                data[:, c] = vitals_means[c]
                masked[:,c] = 1
        
        return data,masked
    
    def statics_augment(self, data):
        masked = np.zeros_like(data)
        num_feat = len(data)
        num_corrupt = int(self.corrupt_rate*num_feat)
        corrupt_idxs = np.random.choice(num_feat, size=num_corrupt, replace=False)
        
        # NOTE again these are hardcoded since we have random data.
        lab_means = 0
        lab_stds = 1

        for i in corrupt_idxs:
            data[i] = lab_means
            masked[i] = 1
        
        # add noise
        for i in range(num_feat):
            data[i] += np.random.normal()*lab_stds*0.1
        
        # NOTE if we have categorical features, might want a separate augmentation protocol here. 

        return data, masked


    def __getitem__(self, idx):
        """
        Returns:
            tensors (dict)
        """
        
        tensors = {}

        ###################################################################################################
        # NOTE: this __getitem__ function is just a placeholder.
        # It dynamically generates random trajectory-structured data on each call.
        # In reality, the raw data would be loaded from some actual filepath on a given call, and then
        # would be augmented and processed for use in the model. 
        ###################################################################################################
        

        ########################################################################################
        # Generate some random trajectory data
        ########################################################################################

        # Generate a random medical record number (MRN)
        mrn = str(np.random.choice(100))

        RAW_TIMESERIES_LENGTH = 12

        # Random static structured data (d in the paper)
        STRUC_FEATS = 10
        struc_data = np.random.normal(size=STRUC_FEATS)

        # Random structured data timeseries (w in the paper)
        # Called vitals here since vitals data is typically a structured timeseries
        VITALS_FEATS = 10
        # Raw structured data timeseries is a dictionary with keys equal to the measurement times and values with the vectors
        vitals_data = {i:np.random.normal(size=VITALS_FEATS) for i in range(RAW_TIMESERIES_LENGTH)}

        # Random waveform generation
        # The raw data is a dictionary keys equal to the measurement times and values being another dictionary of {channel_name:samples}
        # For this simple example, treat the number of channels as 1 for simplicity -- we let this be Lead II of an ECG, sampled at 125 Hz for 30s
        wf_data = {i: {'leadII':np.random.normal(size=125*30)} for i in range(RAW_TIMESERIES_LENGTH)}
        
        times = list(vitals_data.keys())


        ########################################################################################
        # Generate batches to pass into the model 
        ########################################################################################

        # During Self Sup learning, we sample a random contiguous chunk to use as a batch in model training. 
        if self.task =='ssl':
            start_time = np.random.randint(0, 1+len(times)-self.min_seq_len)
            traj_len = np.random.randint(self.min_seq_len, self.max_seq_len+1)
            end_time = min(start_time + traj_len, len(times))

            
            #### Make trajectory-structured data from the raw data store
            ret_signals_traj = self.make_signals_traj(wf_data, start_time, end_time, index=0)
            ret_signals_traj_aug = self.make_signals_traj(wf_data, start_time, end_time, index=1)

            ret_structured_data = self.make_structured_data_traj(vitals_data, start_time, end_time)
            ret_structured_data_aug = self.make_structured_data_traj(vitals_data, start_time, end_time)

            ret_statics = self.make_static_data(struc_data)
            ret_statics_aug = self.make_static_data(struc_data)


            ################################################################################# 
            # Do augmentation to generate multiple views of the different data modalities
            #################################################################################

            ### First the waveforms
            tensors['signals_timeseries1'], _ = self.mask_augmentation(ret_signals_traj)
            tensors['signals_timeseries2'], _ = self.mask_augmentation(ret_signals_traj_aug)

            ### Now structured data timeseries            
            ret_structured_data, _ = self.struct_timeseries_augment(ret_structured_data)
            ret_structured_data_aug, _ = self.struct_timeseries_augment(ret_structured_data_aug)
            # NOTE: maybe want some postproc, e.g., normalization, here.
            tensors['structured_timeseries1'] = ret_structured_data
            tensors['structured_timeseries2'] = ret_structured_data_aug

            ### Finally, statics
            ret_statics, _ = self.statics_augment(ret_statics)
            ret_statics_aug, _ = self.statics_augment(ret_statics_aug)
            # NOTE: maybe want some postproc, e.g., normalization, here.
            tensors['statics1'] = ret_statics
            tensors['statics2'] = ret_statics_aug
        

        # For supervised finetuning, we generate a labelled batch 
        elif self.task == 'example_task':
            start_time = 0
            end_time = min(start_time + self.eval_seq_len, len(times))
            ret_signals_traj = self.make_signals_traj(wf_data, start_time, end_time)
            ret_structured_data = self.make_structured_data_traj(vitals_data, start_time, end_time)
            ret_statics = self.make_static_data(struc_data)
            label = np.random.choice(2)
            tensors['example_task'] = label
        else:
            raise NotImplementedError(f'Still need to implement dataloading for {self.task} task!')


        tensors['signals_timeseries'] = ret_signals_traj
        tensors['pt_ids'] = mrn 
        tensors['start_times'] = times[0]
        tensors['structured_timeseries'] = ret_structured_data
        tensors['statics'] = ret_statics

        # Log the actual seq end idx because we may have ones of different lengths. 
        tensors['end_idx'] = (end_time-start_time)-1


        return tensors

