#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import os
import subprocess

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.nn.modules.loss import _WeightedLoss
import numpy as np


class Mixup_dataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform='mix', imputation=None, 
    noise=0.01, n_choise=10, dropout=0.4, device=torch.device("cpu")):
        self.transform = transform
        self.imputation = imputation
        self.data_num = data.shape[0]
        self.data = data
        self.label = label
        self.channel = label.shape[1]
        self.n_choise = n_choise
        self.noise = 1/noise
        self.dropout = np.random.uniform(dropout)
        self.device = device
        with np.errstate(invalid='ignore'):
            self.p = np.matmul(np.minimum(1 / label.sum(axis=0), np.ones(label.shape[1])),
                           label.T)
        self.p /= self.p.sum()


    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform == 'mix':
#             idx_rand = torch.multinomial(torch.ones(self.data.shape[0]), self.n_choise)
#             idx_rand = torch.multinomial(torch.ones(self.data.shape[0]), np.random.randint(1, self.n_choise))
            idx_rand = np.random.choice(range(self.data_num), 
                                        size=np.random.randint(1, self.n_choise), p=self.p)
            out_data = self.data[idx_rand].mean(axis=0)
            out_label = self.label[idx_rand].mean(axis=0)

        else:
            out_data = self.data[idx]
            out_label =  self.label[idx]
            
        if self.noise:
            with np.errstate(invalid='ignore'):
                out_data = np.random.beta(self.noise * out_data + 1, self.noise * (1-out_data) + 1)
        
        if self.dropout:
            idx_drop = np.random.choice(out_data.shape[0], size=int(out_data.shape[0]*self.dropout), 
                                        replace=False)
            out_data[idx_drop] = np.nan
        
        if self.imputation:
            out_data = self.imputation.transform(out_data.reshape((1,out_data.shape[0]))
                                                ).reshape(out_data.shape[0])
            
        out_data = torch.from_numpy(out_data).to(self.device).float()
        out_label = torch.from_numpy(out_label).to(self.device)
        return out_data, out_label
    

class OneHotCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss
    
    
class SmoothCrossEntropy(nn.Module):
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05,one_hotted=False):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.one_hotted = one_hotted
    def forward(self, x, target):
        if self.training:
            x = x.float()
            if self.one_hotted!=True:
                target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            if self.one_hotted!=True:
                loss = F.cross_entropy(x, target.long())
            else:
                loss = OneHotCrossEntropy(x, target)
            return loss
        

# https://github.com/Bjarten/early-stopping-pytorch modified
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def detect_delim(f_input):
    df_csv = pd.read_csv(f_input, nrows=3)
    df_tsv = pd.read_csv(f_input, sep='\t', nrows=3)
    if df_csv.shape[1] > df_tsv.shape[1]:
        return 'csv'
    else:
        return 'tsv'


def check_tile(t, ref, p_bedtools):
    if not os.path.exists('{d}/../data/{r}.win{t}.bed.gz'.format(r=ref, d=os.path.dirname(os.path.abspath(__file__)), t=t)):
        cmd = '{b} makewindows -g {d}/../data/{r}.sort.genome -w {t} > {d}/../data/{r}.win{t}.bed'.format(
            b=p_bedtools, d=os.path.dirname(os.path.abspath(__file__)), t=t, r=ref)
        print(cmd)
        subprocess.run(cmd, shell=True)

        cmd = 'gzip {d}/../data/{r}.win{t}.bed'.format(r=ref, d=os.path.dirname(os.path.abspath(__file__)), t=t)
        print(cmd)
        subprocess.run(cmd, shell=True)


def parse_bismark(f, t):
    df = pd.read_csv(f, sep='\t', header=None)
    df.columns = ['chromosome', 'start', 'end', 'methylated_frequency', 'meth', 'deme']
    if t == 0:
        df['CpGs'] = df['chromosome'] + ':' + (df['start']).astype(str)
    else:
        df['CpGs'] = df['chromosome'] + ':' + (df['start']).astype(str) + '-' + (df['end']).astype(str)
    
    if df['methylated_frequency'].max() > 1:
        df['methylated_frequency'] /= 100   
    # df = df[df[['meth', 'deme']].sum(axis=1) > th_cov]
    # df['methylated_frequency'] = (df['meth'] + 1) / (df['meth'] + df['deme'] + 2)
    df = df[['CpGs', 'methylated_frequency']]
    return df


def tile_bismark(f_bismark, tile_bp, p_bedtools):

    ref = 'hg38'

    n = f_bismark.split('/')[-1].split('.bis')[0]
    print('sample name : ', n)
    os.makedirs('tmp_menet', exist_ok=True)
    check_tile(tile_bp, ref, p_bedtools)

    cmd = '{b} sort -i {f} > {f_sort}'.format(b=p_bedtools, f=f_bismark, f_sort='tmp_menet/tmp.sort.txt')
    print(cmd)
    subprocess.run(cmd, shell=True)

    cmd = '{b} map -a {d}/../data/{r}.win{x}.bed.gz -b {bis} -c 4,5,6 -o mean,sum,sum | grep -v "\.\s*\." > {o}'.format(
        b = p_bedtools, d=os.path.dirname(os.path.abspath(__file__)),
        x = tile_bp, bis='tmp_menet/tmp.sort.txt', o='tmp_menet/tmp.tile.txt', r=ref)
    print(cmd)
    subprocess.run(cmd, shell=True)
    f = 'tmp_menet/tmp.tile.txt'
    df = parse_bismark(f, tile_bp)
    df.columns = ['CpGs', n]
    df.index = df['CpGs']
    df.index.name = 'CpGs'
    df = df[[n]]

    shutil.rmtree('tmp_menet')
    
    return df


def tile_array(f_input, input_filetype, tile_bp):

    ref = 'hg38'

    try:
        df_probeid_name = pd.read_csv(
            '{d}/../data/probeID_name_win{t}bp.{r}.csv.gz'.format(
                r=ref, d=os.path.dirname(os.path.abspath(__file__)), t=tile_bp), index_col=0)
    except ValueError:
        print('The index file; ProbeID - tiling windows is not prepared for the tile(bp).')

    if input_filetype == 'tsv':
        df_array = pd.read_csv(f_input, comment="!", sep='\t', index_col = 0)
    else:
        df_array = pd.read_csv(f_input, index_col=0)

    col_samples = df_array.columns

    df_array['probeID'] = df_array.index
    df_array = pd.merge(df_array, df_probeid_name, how='left', on='probeID')
    df_array = df_array.groupby(by='name').mean()
    df_array = df_array[col_samples]
    
    return df_array