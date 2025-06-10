#!/usr/bin/env python
# coding=utf-8

from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test, Data_eval, Data_WV2_8, Data_new
from torchvision import transforms
import torch, numpy  #h5py, 
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    cfg = cfg
    if cfg['data']['n_colors'] == 4:
        if cfg['dataset'] == 'QB_remake':
            return Data_new(data_dir_ms, data_dir_pan, cfg, transform=transform())
        else :
            return Data(data_dir_ms, data_dir_pan, cfg, transform=transform())
    else :
        return Data_WV2_8(data_dir_ms, data_dir_pan, cfg, transform=transform())
    
def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    cfg = cfg
    if cfg['data']['n_colors'] == 4:
        return Data_test(data_dir_ms, data_dir_pan, cfg, transform=transform())
    else :
        return Data_WV2_8(data_dir_ms, data_dir_pan, cfg, transform=transform())

def get_eval_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    cfg = cfg
    return Data_eval(data_dir_ms, data_dir_pan, cfg, transform=transform())