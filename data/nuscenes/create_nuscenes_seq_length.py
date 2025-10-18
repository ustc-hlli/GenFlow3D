# -*- coding: utf-8 -*-
'''
Created on Tue Jul  2 23:37:52 2024

@author: lihl
'''

import os
import os.path as osp

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import warnings
import h5py

from labelmap import get_label_map_from_file
from unsup_flow.datasets.nuscenes.nuscenes_parser import NuScenesParser
from usfl_io.io_tools import nusc_add_nn_segmentation_flow_for_t1
from npimgtools import Transform

def process_pc_seq(cur_sample, next_sample, nusc: NuScenesParser, seq_length: int, save_dict: dict):
    dataset_freq = 20.0
    
    scene = nusc.get('scene', cur_sample['scene_token'])
    sd_toks = nusc.get_token_list('sample_data', cur_sample['data']['LIDAR_TOP'], recurse_by=-1)
    cur_sd_idx = sd_toks.index(cur_sample['data']['LIDAR_TOP'])
    if(next_sample is not None):
        next_sd_idx = sd_toks.index(next_sample['data']['LIDAR_TOP'])
    else:
        next_sd_idx = len(sd_toks)
    if(next_sd_idx - cur_sd_idx < seq_length):
        print('not enough frame number: %d' % (next_sd_idx - cur_sd_idx))
        return 'not_enough_frame_number'
    
    nusc2carla_labelmap = get_label_map_from_file('nuscenes', 'nuscenes2carla')
    nusc2statdynground_labelmap = get_label_map_from_file(
        'nuscenes', 'nuscenes2static_dynamic_ground'
    )
    
    # check the timestamps
    timestamps = [nusc.get('sample_data', sd_toks[i])['timestamp'] for i in range(cur_sd_idx, cur_sd_idx + seq_length)]
    for t0, t1 in zip(timestamps[:-1], timestamps[1:]):
        frame_diff_0_1 = (t1 - t0) / (1e6 / dataset_freq)
        frame_diff_0_1_int = int(round(frame_diff_0_1))
        if(frame_diff_0_1_int != 1 or 
           np.abs(frame_diff_0_1 - frame_diff_0_1_int) > 0.3 or
           not np.allclose(t1 - t0, 1e6 / dataset_freq, rtol=0.1, atol=5000)):
            warn_mes = 'irregular time difference: %f s, frequency: %f Hz' % ((t1 - t0) / 1e6, dataset_freq)
            tqdm.write(cur_sample['token'] + warn_mes)
            return 'irregular_temporal_sampling'
        #if(frame_diff_0_1_int > 1):
        #    assert not np.allclose(
        #        t1 - t0, 1e6 / dataset_freq, rtol=0.1, atol=5000
        #    ), (
        #        t1 - t0,
        #        cur_sample['token'],
        #    )
        #    tqdm.write(cur_sample['token'] + ': missing sample data for correct frequency')
        #    return 'irregular_temporal_sampling'
    
    # get poses of dynamic objects in the global coordinate
    instances = {}
    instance_idx = 1
    for ann_tok in cur_sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        if(ann['next'] != ''):
            next_ann = nusc.get('sample_annotation', ann['next'])
            if((np.array(ann['size'])[[1, 0, 2]] != np.array(next_ann['size'])[[1, 0, 2]]).any()):
                print('Changed object size.')
                return 'Changed_object_size'
        size = np.array(ann['size'])[[1, 0, 2]]
        instance = nusc.get('instance', ann['instance_token'])
        if(ann['category_name'] not in nusc2statdynground_labelmap.mname_rnames_dict['dynamic']):
            continue
        try:
            trajectory = nusc.get_interpolated_instance_poses__m(instance, timestamps)
        except AssertionError:
            print('Interpolation failed!')
            return 'pose_interpolation_failed'
        
        instances[str(instance_idx)] = {'size': size,
                                        'poses': trajectory}
        instance_idx += 1
    
    if(next_sample is not None):
        for ann_tok in next_sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            if(ann['prev'] != ''):
                continue
            size = np.array(ann['size'])[[1, 0, 2]]
            instance = nusc.get('instance', ann['instance_token'])
            if(ann['category_name'] not in nusc2statdynground_labelmap.mname_rnames_dict['dynamic']):
                continue
            try:
                trajectory = nusc.get_interpolated_instance_poses__m(instance, timestamps)
            except AssertionError:
                print('Interpolation failed!')
                return 'pose_interpolation_failed'
        
            instances[str(instance_idx)] = {'size': size.astype('float32'),
                                            'poses': trajectory}
            instance_idx += 1
    
    # get point clouds
    pcs = {}
    for i in range(cur_sd_idx, cur_sd_idx + seq_length):
        sd = nusc.get('sample_data', sd_toks[i])
        try:
            pcl, ego_mask = nusc.get_pointcloud(sd, ref_frame='ego')
            unmasked_pcl, _ = nusc.get_pointcloud(sd, ref_frame='ego', remove_ego_points=False)
        except AttributeError:
            print('Failed to load point clouds due to missing ego_points_decisions')
            return 'missing_ego_points_decisions'
        
        pcl = pcl.T[:, :3]
        ego_pose = nusc.get_ego_pose_at_timestamp(cur_sample['scene_token'], sd['timestamp'])
        
        if(i < cur_sd_idx + seq_length - 1):
            new_sd = nusc.get('sample_data', sd_toks[i+1])
            new_ego_pose = nusc.get_ego_pose_at_timestamp(cur_sample['scene_token'], new_sd['timestamp'])
            odom = new_ego_pose.copy().invert() * ego_pose
            
            flow = pcl @ (odom.rot_mat() - np.eye(3)).T + odom.trans()
        
        # dynamic masks
        object_mask = -np.ones([pcl.shape[0], 1], dtype='int32')
        for instance_idx in instances.keys():
            instance = instances[instance_idx]
            size = instance['size']
            pose = instance['poses'][i-cur_sd_idx]
            
            instance_pose_ego = ego_pose.copy().invert() * pose
            inv_instance_pose_ego = instance_pose_ego.copy().invert()
            pcl_instance = pcl @ inv_instance_pose_ego.rot_mat().T + inv_instance_pose_ego.trans()
            cur_instance_mask = np.logical_and((np.abs(pcl_instance) < size / 2.0).all(axis=-1), pcl[:, 2] > 0.0)
            object_mask[cur_instance_mask] = int(instance_idx)
            
            # transform the instance poses into the ego coordinate
            instances[instance_idx]['poses'][i-cur_sd_idx] = instance_pose_ego
            
            if(i < cur_sd_idx + seq_length - 1):
                new_pose = instance['poses'][i-cur_sd_idx+1]
                new_instance_pose_ego = new_ego_pose.copy().invert() * new_pose
                dyn_flow_transf = new_instance_pose_ego * inv_instance_pose_ego
                dyn_flow = pcl @ (dyn_flow_transf.rot_mat() - np.eye(3)).T + dyn_flow_transf.trans()
                flow[cur_instance_mask] = dyn_flow[cur_instance_mask]
        
        if(i < cur_sd_idx + seq_length - 1):
            pcs[str(i-cur_sd_idx)] = {'pts': pcl.astype('float32'),
                                  'instance_mask': object_mask,
                                  'flow': flow.astype('float32'),
                                  'ego_pose': ego_pose.as_htm().astype('float64')}
        else:
            pcs[str(i-cur_sd_idx)] = {'pts': pcl.astype('float32'),
                                  'instance_mask': object_mask,
                                  'ego_pose': ego_pose.as_htm().astype('float64')}
        
    for instance in instances.values():
        instance['poses'] = np.stack([transf.as_htm() for transf in instance['poses']], axis=0).astype('float64')
        
    save_dict['pcs'] = pcs
    save_dict['instances'] = instances
    
    return 'fine'


def main(path_out: str, nusc_root: str, version: str):
    nusc = NuScenesParser(
        version=version,
        dataroot=nusc_root,
        verbose=True,
    )
    results = {}
    count_results = {}
    
    for scene in nusc.scene:
        print('processing ' + scene['name'])
        
        # get all sample tokens in this sequence
        first_sample_tok = scene['first_sample_token']
        sample_tokens = nusc.get_token_list('sample', first_sample_tok, recurse_by=-1)
        
        # process the samples
        key_ind = range(0, len(sample_tokens))
        for _, cur_sample_idx in tqdm(enumerate(key_ind)):
            save_dict = {}
            cur_sample = nusc.get('sample', sample_tokens[cur_sample_idx])
            if(cur_sample_idx < len(sample_tokens) - 1):
                next_sample = nusc.get('sample', sample_tokens[cur_sample_idx+1])
            else:
                next_sample = None
            
            cur_result = process_pc_seq(cur_sample, next_sample, nusc, length, save_dict)
            if(cur_result != 'fine'):
                results[cur_sample['token']] = cur_result
                if(cur_result not in count_results):
                    count_results[cur_result] = 0
                count_results[cur_result] += 1
                tqdm.write(str(count_results))
                continue
            
            # save the sample (as h5 files)
            cur_path = os.path.join(path_out, scene['name'], str(cur_sample_idx))
            if(not os.path.exists(cur_path)):
                os.makedirs(cur_path)
            filename = "sequence.h5" 
            filename = os.path.join(cur_path, filename)
            with h5py.File(filename, 'w') as f:
                for k, v in save_dict.items(): # 'pcs', 'instances'
                    cur_group = f.create_group(k)
                    for kn, vn in v.items(): # '0', '1', '2' ...
                        cur_obj = cur_group.create_group(kn)
                        for ka, va in vn.items(): # 'pts', 'ego_pose', 'flow', 'poses' ...
                            cur_obj[ka] = va 
            
            results[cur_sample["token"]] = cur_result
            if(cur_result not in count_results):
                count_results[cur_result] = 0
            count_results[cur_result] += 1
            tqdm.write(str(count_results))  
            
            
    return None



# global settings
length = 10
path_out = #your nuscenes raw data path 
nusc_root = #your proessed nuscenes data output path

main(
     path_out,
     nusc_root,
     version='v1.0-trainval' 
)