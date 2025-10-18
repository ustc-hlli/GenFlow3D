# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:41:11 2024

@author: lihl
"""

import numpy as np
import h5py
import pickle
import os
import torch
from torch.utils.data import Dataset

import utils

def get_valid_points(pc):
    
    remote_mask = np.logical_and(np.abs(pc[:, 0]) < 35.0, np.abs(pc[:, 1]) < 35.0)
    ground_mask = (pc[:, 2] > 0.25)
    
    mask = np.logical_and(remote_mask, ground_mask)
    
    return mask

def get_valid_points_av2(pc):
    points = pc['pts'][:, :3]
    
    remote_mask = np.logical_and(np.abs(points[:, 0]) < 35.0, np.abs(points[:, 1]) < 35.0)
    ground_mask = ~(pc['is_ground'][:])
    
    mask = np.logical_and(remote_mask, ground_mask)
    
    return mask

class NuScenes_seq(Dataset):
    def __init__(self, root, npoint, length_pre, length_fut, train, return_fut_pc=False):
        assert length_pre > 1
        assert (length_fut + length_pre) <= 10
        
        self.root = os.path.join(root, 'NuScenes_processed_seq')
        self.npoint = npoint
        self.length_pre = length_pre
        self.length_fut = length_fut
        self.train = train
        self.return_fut_pc = return_fut_pc
        
        self.samples = self.get_samples()
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        with h5py.File(os.path.join(sample, 'sequence.h5'), 'r') as f:
            pcs = f['pcs']
            instances = f['instances']
            
            xyzs = []
            feats = []
            flows = []
            fut_xyzs = []
            for pc_idx in range(self.length_pre):
                cur_xyz = pcs[str(pc_idx)]['pts'][:, :3]                
                
                cur_mask = get_valid_points(cur_xyz)
                cur_xyz = cur_xyz[cur_mask, :]                
                
                r = (cur_xyz.shape[0] < self.npoint)
                cur_idx = np.random.choice(cur_xyz.shape[0], self.npoint, replace=r)
                cur_xyz = cur_xyz[cur_idx, :]
                
                xyzs.append(cur_xyz.astype('float32'))
                feats.append(cur_xyz.astype('float32'))
                
                if(pc_idx < self.length_pre-1):
                    cur_flow = pcs[str(pc_idx)]['flow'][:, :3]
                    cur_flow = cur_flow[cur_mask, :]
                    cur_flow = cur_flow[cur_idx, :]
                    flows.append(cur_flow.astype('float32'))
                
                if(pc_idx == self.length_pre-1):
                    last_mask = cur_mask
                    last_idx = cur_idx
                    
            if(self.return_fut_pc):
                for pc_idx in range(self.length_pre, self.length_pre + self.length_fut):
                    cur_xyz = pcs[str(pc_idx)]['pts'][:, :3]
                    cur_mask = get_valid_points(cur_xyz)
                    cur_xyz = cur_xyz[cur_mask, :]                
                
                    r = (cur_xyz.shape[0] < self.npoint)
                    cur_idx = np.random.choice(cur_xyz.shape[0], self.npoint, replace=r)
                    cur_xyz = cur_xyz[cur_idx, :]

                    fut_xyzs.append(cur_xyz.astype('float32'))  
                    
            pre_flow = 0
            for pc_idx in range(self.length_pre, self.length_pre + self.length_fut):
                gener_flow = utils.compute_flow_two_frame(pcs, instances, self.length_pre - 1, pc_idx)
                cur_flow = gener_flow - pre_flow
                cur_flow = cur_flow[last_mask, :]
                cur_flow = cur_flow[last_idx, :]
                flows.append(cur_flow.astype('float32'))
                
                pre_flow = gener_flow

        if(self.return_fut_pc):
            return xyzs, feats, flows, fut_xyzs
        else:    
            return xyzs, feats, flows
        
    def __len__(self):
        return len(self.samples)
    
    def get_samples(self):
        #scenes = os.listdir(self.root)
        #scenes.sort()
        #assert(len(scenes) == 850)
        
        #scenes_train = scenes[:650]
        #scenes_eval = scenes[-200:]

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nuscenes_split', 'train.pkl'), 'rb') as f:
            scenes_train = pickle.load(f)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nuscenes_split', 'eval.pkl'), 'rb') as f:
            scenes_eval = pickle.load(f)

        samples = []
        if(self.train):
            for s in scenes_train:
                for cur_root, cur_dirs, cur_files in os.walk(os.path.join(self.root, s)):
                    if(len(cur_dirs) == 0):
                        samples.append(cur_root)
            assert len(samples) == 22594
        else:
            for s in scenes_eval:
                for cur_root, cur_dirs, cur_files in os.walk(os.path.join(self.root, s)):
                    if(len(cur_dirs) == 0):
                        samples.append(cur_root)
            assert len(samples) == 7130

        return samples
    
class AV2(Dataset):
    def __init__(self, root, npoint, length_pre, length_fut, train, return_fut_pc=False):
        assert length_pre > 1
        assert (length_fut + length_pre) <= 10
        
        self.root = os.path.join(root, 'av2_processed')
        self.npoint = npoint
        self.length_pre = length_pre
        self.length_fut = length_fut
        self.train = train
        self.return_fut_pc = return_fut_pc
        
        self.samples = self.get_samples()
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        with h5py.File(os.path.join(sample, 'sequence.h5'), 'r') as f:
            pcs = f['pcs']
            instances = f['instances']
            
            xyzs = []
            feats = []
            flows = []
            valids = []
            fut_xyzs = []
            for pc_idx in range(self.length_pre):
                cur_xyz = pcs[str(pc_idx)]['pts'][:, :3]
                
                cur_mask = get_valid_points_av2(pcs[str(pc_idx)])
                cur_xyz = cur_xyz[cur_mask, :]
                
                r = (cur_xyz.shape[0] < self.npoint)
                cur_idx = np.random.choice(cur_xyz.shape[0], self.npoint, replace=r)
                cur_xyz = cur_xyz[cur_idx, :]
                
                xyzs.append(cur_xyz.astype('float32'))
                feats.append(cur_xyz.astype('float32'))
                
                if(pc_idx < self.length_pre-1):
                    cur_flow = pcs[str(pc_idx)]['flow'][:, :3]
                    cur_flow = cur_flow[cur_mask, :]
                    cur_flow = cur_flow[cur_idx, :]
                    
                    cur_valid = pcs[str(pc_idx)]['valid_mask'][:]
                    cur_valid = cur_valid[cur_mask]
                    cur_valid = cur_valid[cur_idx]
                    cur_valid = cur_valid[:, None] #[N, 1]
                    
                    cur_flow = cur_flow * cur_valid
                    
                    flows.append(cur_flow.astype('float32'))
                    valids.append(cur_valid.astype('float32'))
                    
                if(pc_idx == self.length_pre-1):
                    last_mask = cur_mask
                    last_idx = cur_idx
                    
            if(self.return_fut_pc):
                for pc_idx in range(self.length_pre, self.length_pre + self.length_fut):
                    cur_xyz = pcs[str(pc_idx)]['pts'][:, :3]
                    cur_mask = get_valid_points_av2(pcs[str(pc_idx)])
                    cur_xyz = cur_xyz[cur_mask, :]                
                
                    r = (cur_xyz.shape[0] < self.npoint)
                    cur_idx = np.random.choice(cur_xyz.shape[0], self.npoint, replace=r)
                    cur_xyz = cur_xyz[cur_idx, :]

                    fut_xyzs.append(cur_xyz.astype('float32')) 
                    
            pre_flow = 0
            pre_valid = np.ones([pcs[str(self.length_pre - 1)]['valid_mask'][:].shape[0], 1]).astype('bool')
            for pc_idx in range(self.length_pre, self.length_pre + self.length_fut):
                gener_flow, gener_valid = utils.compute_flow_two_frame_av2(pcs, instances, self.length_pre - 1, pc_idx)
                cur_flow = gener_flow - pre_flow
                cur_flow = cur_flow[last_mask, :]
                cur_flow = cur_flow[last_idx, :]
                
                cur_valid = np.logical_and(gener_valid, pre_valid)
                cur_valid = cur_valid[last_mask, :]
                cur_valid = cur_valid[last_idx, :]
                
                cur_flow = cur_flow * cur_valid
                
                flows.append(cur_flow.astype('float32'))
                valids.append(cur_valid.astype('float32'))
                
                pre_flow = gener_flow
                pre_valid = gener_valid
        
        if(self.return_fut_pc):
            return xyzs, feats, flows, valids, fut_xyzs
        else:    
            return xyzs, feats, flows, valids

        
    def __len__(self):
        return len(self.samples)
        
    def get_samples(self):
        if(self.train):
            split = 'train'
        else:
            split = 'val'
        scenes = os.listdir(os.path.join(self.root, split))
        scenes.sort()
        assert ((len(scenes) == 700 and self.train) or (len(scenes) == 150 and (not self.train)))
        
        samples = []
        for s in scenes:
            for cur_root, cur_dirs, cur_filse in os.walk(os.path.join(self.root, split, s)):
                if(len(cur_dirs) == 0):
                    samples.append(cur_root)
        if(self.train):
            assert len(samples) == 10540
        else:
            assert len(samples) == 2254

        return samples 