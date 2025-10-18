# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:51:41 2024

@author: lihl
"""

import av2
import av2._r as rust
from av2.map.map_api import ArgoverseStaticMap
from av2.torch.structures.flow import BOUNDING_BOX_EXPANSION, Flow, cuboids_to_id_cuboid_map
from av2.torch.structures.sweep import Sweep
from av2.evaluation.scene_flow.constants import (
    CATEGORY_TO_INDEX,
    SCENE_FLOW_DYNAMIC_THRESHOLD,
)

import torch
import numpy as np
import pandas as pd
import os
import h5py
from pathlib import Path
from tqdm import tqdm
import sys
import argparse


def index_map(file_index, nprev, nfuture):
    N = file_index.shape[0]
    indices = {}
    for i in range(N):
        if(i >= nprev-1 and i+nfuture < N):
            future_log_id = file_index.loc[i+nfuture, 'log_id']
            current_log_id = file_index.loc[i, 'log_id']
            previous_log_id = file_index.loc[i-nprev+1, 'log_id']
            if((current_log_id == future_log_id) and (current_log_id == previous_log_id)):
                if(current_log_id not in indices.keys()):
                    indices[current_log_id] = []
                indices[current_log_id].append(i)
                
    return indices




parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True,
                    help='Path to the argoverse 2 dataset')
parser.add_argument('--out_root', required=True,
                    help='Output path')
parser.add_argument('--split_name', required=True,
                    help='Split to process (train or val)')
parser.add_argument('--nframe_bt_samples',
                    help='',
                    default=10)
parser.add_argument('--nframe_future',
                    help='',
                    default=10)
par_arg = parser.parse_args()

nframe_bt_samples = par_arg.nframe_bt_samples
nframe_future = par_arg.nframe_future

#data_root = # your av2 raw data path 
#split_name = 'train' # 'train' or 'val'
#out_root = # your av2 processed data output path

data_path = os.path.join(par_arg.data_root, 'av2', 'sensor', par_arg.split_name)
out_path = os.path.join(par_arg.out_root, par_arg.split_name)

backend = rust.DataLoader(par_arg.data_root, 'av2', 'sensor', par_arg.split_name, 1, False) # av2 official default 
file_index = backend.file_index.to_pandas()
print(file_index.shape)

results = {}
count_results = {}
indices = index_map(file_index, 1, nframe_future)
log_idx = 0
for log_id, idx_list in indices.items():

    print('processing log: ', log_id)
    key_inds = range(0, len(idx_list), nframe_bt_samples) 
    for _, i in tqdm(enumerate(key_inds)):
        save_dict = {}
        flag = 'fine'
        
        backend_index = idx_list[i]
        log = str(file_index.loc[backend_index, 'log_id'])
        log_map_dirpath = Path(os.path.join(data_path, log, 'map'))
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
                
        pcs = {}
        instances = {}
        instance_idx = 1
        for j in range(nframe_future):
            # get point clouds
            future_log = str(file_index.loc[backend_index+j, 'log_id'])
            if(future_log != log):
                flag = 'wrong_log_id'
                break
            future_sweep = Sweep.from_rust(backend.get(backend_index+j), avm=avm)
            future_ego = future_sweep.city_SE3_ego.matrix().squeeze(0)
            future_pcl = future_sweep.lidar.as_tensor()[:, :3]
            if(future_sweep.cuboids is None):
                print('No oject box!')
                flag = 'no_object_box'
                break
            future_cuboids = future_sweep.cuboids
            future_cuboids_map = cuboids_to_id_cuboid_map(future_cuboids)
            cat_inds = torch.zeros(len(future_pcl), dtype=torch.int32)
            
            # get the dynamic objects.            
            for id in future_cuboids_map:
                if(id not in instances.keys()):
                    instances[id] = {'poses': np.zeros([nframe_future, 4, 4]), 'size': np.zeros([1, 3]), 'idx': instance_idx}
                    instance_idx += 1
                c0 = future_cuboids_map[id]
                c0.length_m += BOUNDING_BOX_EXPANSION
                c0.width_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(future_pcl.cpu().numpy())
                obj_pts, obj_mask = torch.as_tensor(obj_pts, dtype=torch.float32), torch.as_tensor(obj_mask)
                #cat_inds[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
                cat_inds[obj_mask] = instances[id]['idx']
                
                c0_size = c0.dims_lwh_m
                c0_pose = c0.dst_SE3_object.transform_matrix
                if((instances[id]['size'] != np.zeros([1, 3])).any() and 
                   (c0_size != np.zeros([1, 3])).any() and 
                   (c0_size != instances[id]['size']).any()):
                    print('Changed object size!')
                    flag = 'changed_object_size'
                    break
                 
                instances[id]['size'] = c0_size
                instances[id]['poses'][j] = c0_pose
            if(flag != 'fine'):
                break
            
            # compute the scene flow
            if(j < nframe_future-1):
                next_log = str(file_index.loc[backend_index+j+1, 'log_id'])
                if(next_log != log):
                    flag = 'wrong_log_id'
                    break
                next_sweep = Sweep.from_rust(backend.get(backend_index+j+1), avm=avm)
                if(next_sweep.cuboids is None):
                    print('No oject box!')
                    flag = 'no_object_box'
                    break
                flow = Flow.from_sweep_pair((future_sweep, next_sweep))
                
            if(j < nframe_future-1):
                pcs[str(j)] = {'pts': future_pcl.cpu().numpy().astype('float32'),
                               'instance_mask': cat_inds.cpu().numpy(),
                               'flow': flow.flow.cpu().numpy().astype('float32'),
                               'valid_mask': flow.is_valid.cpu().numpy(),
                               'dynamic_mask': flow.is_dynamic.cpu().numpy(),
                               'ego_pose': future_ego.cpu().numpy().astype('float64'),
                               'is_ground': future_sweep.is_ground.cpu().numpy().astype('bool')}
            else:
                pcs[str(j)] = {'pts': future_pcl.cpu().numpy().astype('float32'),
                               'instance_mask': cat_inds.cpu().numpy(),
                               'ego_pose': future_ego.cpu().numpy().astype('float64'),
                               'is_ground': future_sweep.is_ground.cpu().numpy().astype('bool')}
                
        if(flag != 'fine'):
            results[log + ' idx=' + str(i)] = flag
            if(flag not in count_results):
                count_results[flag] = 0
            count_results[flag] += 1
            tqdm.write(str(count_results)) #print(str(count_results))
            continue
        
        tagged_instances = {}
        for instance_id in instances:
            instace = instances[instance_id]
            tagged_instances[str(instace['idx'])] = {'poses': instace['poses'].astype('float64'),
                                                     'size': instace['size'].astype('float32'),
                                                     'id': instance_id}
        
        # save the sample
        save_dict['pcs'] = pcs
        save_dict['instances'] = tagged_instances
        
        cur_path = os.path.join(out_path, log, str(i))
        if(not os.path.exists(cur_path)):
            os.makedirs(cur_path)
        filename = 'sequence.h5'
        filename = os.path.join(cur_path, filename)
        with h5py.File(filename, 'w') as f:
            for k, v in save_dict.items(): # 'pcs', 'instances'
                cur_group = f.create_group(k)
                for kn, vn in v.items(): # '0', '1', '2' ...
                    cur_obj = cur_group.create_group(kn)
                    for ka, va in vn.items(): # 'pts', 'ego_pose', 'flow', 'poses' ...
                        cur_obj[ka] = va 
                        
        results[log + ' idx=' + str(i)] = flag
        if(flag not in count_results):
            count_results[flag] = 0
        count_results[flag] += 1
        tqdm.write(str(count_results)) #print(str(count_results))
        
    log_idx += 1
    print('%d sequences completed...' % log_idx)
               
print('Completed. Total number of the samples: %d' % count_results['fine'])        