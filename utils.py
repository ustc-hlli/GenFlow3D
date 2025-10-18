# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:18:31 2024

@author: lihl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet2 import pointnet2_utils

KNN_MODE_POINTNET = True
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def cosine_distance(src, dst):
    """
    Calculate cosine similarity distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    src = src / torch.sqrt(torch.sum(src ** 2, -1, keepdim=True) + 1e-8)
    dst = dst / torch.sqrt(torch.sum(dst ** 2, -1, keepdim=True) + 1e-8)
    dist = 1.0 - torch.bmm(src, dst.transpose(1, 2))

    return dist

def computeChamfer(pc1, pc2):
    '''
    pc1: B N 3
    pc2: B M 3
    '''
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim = 2, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2) #[B, N]
    dist2 = dist2.squeeze(1) #[B, M]

    return dist1, dist2

def knn_point(nsample, xyz, new_xyz, pointnet=KNN_MODE_POINTNET):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, K]
    """
    if(pointnet):
        _, group_idx = pointnet2_utils.knn(nsample, new_xyz.contiguous(), xyz.contiguous())
    else:
        sqrdists = square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)

    return group_idx

def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)

    return new_points.permute(0, 2, 1).contiguous()

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

class UpsampleFlow(nn.Module):  
    def forward(self, xyz, sparse_xyz, sparse_flow):
        '''
        3-nn inverse-distance weighted interpolation
        Inputs:
        xyz: coordinates of target points [B, 3, N]
        sparse_xyz: coordinates of source point [B, 3, S]
        
        '''
        if(sparse_flow is None):
            return None
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) #[B, N, 3]
        sparse_xyz = sparse_xyz.permute(0, 2, 1) #[B, S, 3]
        sparse_flow = sparse_flow.permute(0, 2, 1) #[B, S, 3]
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C) #[B, N, 3(S), 3]
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) #[B, N, 3]
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm  #[B, N, 3]

        grouped_flow = index_points_group(sparse_flow, knn_idx)  #[B, N, 3, C]
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2).permute(0, 2, 1) #[B, 3, N]

        return dense_flow 
    
class PointWarping(nn.Module):
    def forward(self, xyz1, xyz2, flow1 = None, neighr=3):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        knn_idx = knn_point(neighr, xyz1_to_2, xyz2) # group flow 1 around points 2
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, neighr, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2
    
def list_index_gather(feats, inds):
    new_feats = []
    for i in range(min(len(feats), len(inds))):
        if(feats[i] is None):
            new_feats.append(None)
            continue
        cur_idx = inds[i] #if(i <= len(inds) - 1) else inds[-1]
        cur_feat = index_points_gather(feats[i].permute(0, 2, 1).contiguous(), cur_idx).permute(0, 2, 1).contiguous()
        new_feats.append(cur_feat)
        
    return new_feats

## numpy functions ##
def compute_metrics(preds, gts):
    # gt: [B, N, 3]
    # pred: [B, 3, N]
    preds = [pred.cpu().numpy() for pred in preds]
    gts = [gt.cpu().numpy() for gt in gts]
    metrics = {'epe': np.zeros([len(gts)]),
               'accr': np.zeros([len(gts)]),
               'accs': np.zeros([len(gts)]),
               'outliers': np.zeros([len(gts)])}
    for i in range(len(gts)): #time
        cur_gt = gts[i].transpose(0, 2, 1) #[B, 3, N]
        cur_pred = preds[i]
        
        cur_gt_mag = np.linalg.norm(cur_gt, ord=2, axis=1) #[B, N]
        cur_err = np.linalg.norm(cur_pred - cur_gt, ord=2, axis=1) #[B, N]
        
        cur_epe = cur_err.mean(axis=1).mean()
        metrics['epe'][i] = cur_epe
        
        cur_accr = np.logical_or(cur_err < 0.1, cur_err/(cur_gt_mag + 1e-8) < 0.1).mean(axis=1).mean()
        metrics['accr'][i] = cur_accr
        
        cur_accs = np.logical_or(cur_err < 0.05, cur_err/(cur_gt_mag + 1e-8) < 0.05).mean(axis=1).mean()
        metrics['accs'][i] = cur_accs
        
        cur_outliers = np.logical_or(cur_err > 0.3, cur_err/(cur_gt_mag + 1e-8) > 0.1).mean(axis=1).mean()
        metrics['outliers'][i] = cur_outliers
        
    return metrics 

def compute_metrics_mask(preds, gts, masks):
    # gt: [B, N, 3]
    # mask: [B, N, 1]
    # pred: [B, 3, N]
    preds = [pred.cpu().numpy() for pred in preds]
    gts = [gt.cpu().numpy() for gt in gts]
    masks = [mask.cpu().numpy() for mask in masks]
    metrics = {'epe': np.zeros([len(gts)]),
               'accr': np.zeros([len(gts)]),
               'accs': np.zeros([len(gts)]),
               'outliers': np.zeros([len(gts)])}
    for i in range(len(gts)): #time
        cur_gt = gts[i].transpose(0, 2, 1) #[B, 3, N]
        cur_mask = masks[i].squeeze(2) #[B, N]
        cur_mask_sum = cur_mask.sum(axis=1) #[B]
        cur_pred = preds[i]
        
        cur_gt_mag = np.linalg.norm(cur_gt, ord=2, axis=1) #[B, N]
        cur_err = np.linalg.norm(cur_pred - cur_gt, ord=2, axis=1) #[B, N]
        
        cur_epe = np.sum(cur_err * cur_mask, axis=1) #[B]
        cur_epe = cur_epe[cur_mask_sum > 0] / cur_mask_sum[cur_mask_sum > 0]
        cur_epe = cur_epe.mean()
        metrics['epe'][i] = cur_epe
        
        cur_accr = np.logical_or(cur_err < 0.1, cur_err/(cur_gt_mag + 1e-8) < 0.1) #[B, N]
        cur_accr = np.sum(cur_accr * cur_mask, axis=1) #[B]
        cur_accr = cur_accr[cur_mask_sum > 0] / cur_mask_sum[cur_mask_sum > 0]
        cur_accr = cur_accr.mean()
        metrics['accr'][i] = cur_accr
        
        cur_accs = np.logical_or(cur_err < 0.05, cur_err/(cur_gt_mag + 1e-8) < 0.05) #[B, N]
        cur_accs = np.sum(cur_accs * cur_mask, axis=1) #[B]
        cur_accs = cur_accs[cur_mask_sum > 0] / cur_mask_sum[cur_mask_sum > 0]
        cur_accs = cur_accs.mean()
        metrics['accs'][i] = cur_accs
        
        cur_outliers = np.logical_or(cur_err > 0.3, cur_err/(cur_gt_mag + 1e-8) > 0.1) #[B, N] 
        cur_outliers = np.sum(cur_outliers * cur_mask, axis=1) #[B]
        cur_outliers = cur_outliers[cur_mask_sum > 0] / cur_mask_sum[cur_mask_sum > 0]
        cur_outliers = cur_outliers.mean()
        metrics['outliers'][i] = cur_outliers
        
    return metrics 

def comput_fut_metrics(flows, last_pc, fut_pcs):
    # flow: [B, 3, N]
    # last_pc: [B, N, 3]
    # fut_pc: [B, N, 3]
    assert last_pc.size()[0] == 1 # B=1
    
    metrics = {'chamfer':np.zeros([len(fut_pcs)])}
    for i in range(len(fut_pcs)):
        cur_flow = flows[i].permute(0, 2, 1) #[B, N, 3]
        cur_fut_pc = fut_pcs[i] #[B, N, 3]

        cur_pred_pc = last_pc + cur_flow #[B, N, 3]        
        
        cur_chamfer1, cur_chamfer2 = computeChamfer(cur_pred_pc, cur_fut_pc) #[B, Nv] [B, M]
        cur_chamfer = (cur_chamfer1.mean(dim=1) + cur_chamfer2.mean(dim=1)).mean()
        metrics['chamfer'][i] = cur_chamfer.cpu().numpy()

        last_pc = last_pc + cur_flow 

    return metrics

def comput_fut_metrics_mask(flows, last_pc, masks, fut_pcs):
    # flow: [B, 3, N]
    # last_pc: [B, N, 3]
    # mask: [B, N, 1]
    # fut_pc: [B, N, 3]
    assert last_pc.size()[0] == 1 # B=1
    
    metrics = {'chamfer':np.zeros([len(fut_pcs)])}
    for i in range(len(fut_pcs)):
        cur_flow = flows[i].permute(0, 2, 1) #[B, N, 3]
        cur_mask = masks[i].squeeze(2).bool() #[B, N]
        cur_mask_sum = cur_mask.sum(axis=1) #[B]
        cur_fut_pc = fut_pcs[i] #[B, N, 3]

        cur_pred_pc = last_pc + cur_flow #[B, N, 3]        
        cur_pred_pc = cur_pred_pc[cur_mask] #[Nv, 3]
        cur_pred_pc = cur_pred_pc.unsqueeze(0) #[1, Nv, 3]
        
        cur_chamfer1, cur_chamfer2 = computeChamfer(cur_pred_pc, cur_fut_pc) #[B, Nv] [B, M]
        cur_chamfer = (cur_chamfer1.mean(dim=1) + cur_chamfer2.mean(dim=1)).mean()
        metrics['chamfer'][i] = cur_chamfer.cpu().numpy()
        
        last_pc = last_pc + cur_flow 
        
    return metrics
    
def compute_flow_two_frame(pcs, instances, src, tgt):
    pts = pcs[str(src)]['pts'][:, :3]
    
    src_ego = pcs[str(src)]['ego_pose'][...]
    tgt_ego = pcs[str(tgt)]['ego_pose'][...]
    src_instance_mask = pcs[str(src)]['instance_mask'][...]
    
    odom = np.linalg.inv(tgt_ego) @ src_ego
    flow = pts @ (odom[:3, :3] - np.eye(3)).T + odom[:3, 3]
    
    for instance_idx in instances.keys():
        instance_mask = (src_instance_mask == int(instance_idx))[:, 0]
        if(instance_mask.sum() == 0):
            continue
        
        instance = instances[instance_idx]
        size = instance['size']
        src_ins_pose = instance['poses'][src]
        tgt_ins_pose = instance['poses'][tgt]
        
        dyn_flow_transf = tgt_ins_pose @ np.linalg.inv(src_ins_pose)
        dyn_flow = pts @ (dyn_flow_transf[:3, :3] - np.eye(3)).T + dyn_flow_transf[:3, 3]
        flow[instance_mask] = dyn_flow[instance_mask]
    
    return flow

def compute_flow_two_frame_av2(pcs, instances, src, tgt):
    if(tgt == src + 1):
        flow = pcs[str(src)]['flow'][:, :3]
        valid = pcs[str(src)]['valid_mask'][:]
        valid = valid[:, None]
        
        return flow, valid 
    else:
        pts = pcs[str(src)]['pts'][:, :3]
    
        src_ego = pcs[str(src)]['ego_pose'][...].astype('float32')
        tgt_ego = pcs[str(tgt)]['ego_pose'][...].astype('float32')
        src_instance_mask = pcs[str(src)]['instance_mask'][...]
    
        odom = np.linalg.inv(tgt_ego) @ src_ego
        odom = odom.astype('float32')
        flow = pts @ (odom[:3, :3] - np.eye(3)).T + odom[:3, 3]
        valid = np.ones([flow.shape[0], 1])
    
        for instance_idx in instances.keys():
            instance_mask = (src_instance_mask == int(instance_idx))
            if(instance_mask.sum() == 0):
                continue
        
            instance = instances[instance_idx]
            size = instance['size']
            src_ins_pose = instance['poses'][src].astype('float32')
            tgt_ins_pose = instance['poses'][tgt].astype('float32')
            if(np.linalg.det(src_ins_pose) == 0 or np.linalg.det(tgt_ins_pose) == 0):
                valid[instance_mask] = 0
            else:
                dyn_flow_transf = tgt_ins_pose @ np.linalg.inv(src_ins_pose)
                dyn_flow_transf = dyn_flow_transf.astype('float32')
                dyn_flow = pts @ (dyn_flow_transf[:3, :3] - np.eye(3)).T + dyn_flow_transf[:3, 3]
                flow[instance_mask] = dyn_flow[instance_mask]
    
        return flow.astype('float32'), valid.astype('bool')