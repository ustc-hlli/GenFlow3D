# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:43:27 2024

@author: lihl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet2 import pointnet2_utils
import layers
import utils
from units import *
        
class GenFlow_rec_seq(nn.Module):
    def __init__(self, npoint):
        super(GenFlow_rec_seq, self).__init__()
        self.npoint = npoint
        self.unit = GenFlow_unit(self.npoint)
        
    def forward(self, pcs, feats, gts):
        assert len(pcs) >= 2
        assert len(gts) >= len(pcs) - 1
        assert len(feats) == len(pcs)
        B = gts[0].size()[0]
        pcs_length = len(pcs)

        pcs = [pc.permute(0, 2, 1).contiguous() for pc in pcs]
        feats = [feat.permute(0, 2, 1).contiguous() for feat in feats]
        gts = [gt.permute(0, 2, 1).contiguous() for gt in gts] if self.training else [None for _ in range(len(gts))]

        
        pcs_cat = torch.cat(pcs, dim=0) 
        feats_cat = torch.cat(feats, dim=0)
        
        ([fps_xyzs1, fps_xyzs2, fps_xyzs3], 
        [fps_inds1, fps_inds2, fps_inds3], 
        [up_feats0, up_feats1, up_feats2, up_feats3]) = self.unit.encode(pcs_cat, feats_cat)
        
        #fps_xyzs4 = list(torch.split(fps_xyzs4, B, dim=0))
        fps_xyzs3 = list(torch.split(fps_xyzs3, B, dim=0))
        fps_xyzs2 = list(torch.split(fps_xyzs2, B, dim=0))
        fps_xyzs1 = list(torch.split(fps_xyzs1, B, dim=0))

        fps_inds1 = list(torch.split(fps_inds1, B, dim=0))
        fps_inds2 = list(torch.split(fps_inds2, B, dim=0))
        fps_inds3 = list(torch.split(fps_inds3, B, dim=0))
        
        up_feats3 = list(torch.split(up_feats3, B, dim=0))
        up_feats2 = list(torch.split(up_feats2, B, dim=0))
        up_feats1 = list(torch.split(up_feats1, B, dim=0))
        up_feats0 = list(torch.split(up_feats0, B, dim=0))
        
        flow_gts1 = utils.list_index_gather(gts, fps_inds1) if self.training else [None for _ in range(len(pcs))]
        flow_gts2 = utils.list_index_gather(flow_gts1, fps_inds2) if self.training else [None for _ in range(len(pcs))]
        flow_gts3 = utils.list_index_gather(flow_gts2, fps_inds3) if self.training else [None for _ in range(len(pcs))]
        flow_preds3, flow_preds2, flow_preds1, flow_preds0 = [], [], [], []
        up_flows3, up_flows2, up_flows1, interp_flows = [], [], [], []
        tot_loss_dif3, tot_loss_dif2, tot_loss_dif1, tot_loss_dif0 = 0, 0, 0, 0
        corr0, corr1, corr2, corr3 = None, None, None, None
        state0, state1, state2, state3 = None, None, None, None
        for idx_gt in range(len(gts)):
            idx_pc = idx_gt + 1
            
            if(self.training and idx_pc > 1 and idx_pc < pcs_length):
                pre_masks = [0, 1]
                mask_pre = pre_masks[np.random.randint(0, 2)] 
            else:
                mask_pre = 1

            future = (idx_pc >= pcs_length)
            ([corr0, corr1, corr2, corr3],
             [state0, state1, state2, state3],
             [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
             [up_flow1, up_flow2, up_flow3, interp_flow],
             [loss_dif0, loss_dif1, loss_dif2, loss_dif3]) = self.unit.decode([gts[idx_pc-1], flow_gts1[idx_pc-1], flow_gts2[idx_pc-1], flow_gts3[idx_pc-1]],
                                                                       [pcs[idx_pc-2], fps_xyzs1[idx_pc-2], fps_xyzs2[idx_pc-2], fps_xyzs3[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       [pcs[idx_pc-1], fps_xyzs1[idx_pc-1], fps_xyzs2[idx_pc-1], fps_xyzs3[idx_pc-1]],
                                                                       [pcs[idx_pc], fps_xyzs1[idx_pc], fps_xyzs2[idx_pc], fps_xyzs3[idx_pc]] if(not future) else [None, None, None, None],
                                                                       [corr0, corr1, corr2, corr3], [state0, state1, state2, state3],
                                                                       [up_feats0[idx_pc-2], up_feats1[idx_pc-2], up_feats2[idx_pc-2], up_feats3[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       [up_feats0[idx_pc-1], up_feats1[idx_pc-1], up_feats2[idx_pc-1], up_feats3[idx_pc-1]],
                                                                       [up_feats0[idx_pc], up_feats1[idx_pc], up_feats2[idx_pc], up_feats3[idx_pc]] if(not future) else [None, None, None, None],
                                                                       [up_flows1[idx_pc-2], up_flows2[idx_pc-2], up_flows3[idx_pc-2], interp_flows[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       [flow_preds0[idx_pc-2], flow_preds1[idx_pc-2], flow_preds2[idx_pc-2], flow_preds3[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       mask_pre)
            if(future):
                corr0, corr1, corr2, corr3 = None, None, None, None
                if(idx_pc < len(gts)):
                    pseudo_pc = (pcs[idx_pc-1] + flow_pred0).detach()
                    pseudo_feat = (pcs[idx_pc-1] + flow_pred0).detach()
                    ([pseudo_fps_xyz1, pseudo_fps_xyz2, pseudo_fps_xyz3], 
                    [pseudo_fps_ind1, pseudo_fps_ind2, pseudo_fps_ind3], 
                    [pseudo_up_feat0, pseudo_up_feat1, pseudo_up_feat2, pseudo_up_feat3]) = self.unit.encode(pseudo_pc, pseudo_feat)

                    pcs.append(pseudo_pc)
                    fps_inds1.append(pseudo_fps_ind1)
                    fps_inds2.append(pseudo_fps_ind2)
                    fps_inds3.append(pseudo_fps_ind3)
                    fps_xyzs1.append(pseudo_fps_xyz1)
                    fps_xyzs2.append(pseudo_fps_xyz2)
                    fps_xyzs3.append(pseudo_fps_xyz3)
                    up_feats0.append(pseudo_up_feat0)
                    up_feats1.append(pseudo_up_feat1)
                    up_feats2.append(pseudo_up_feat2)
                    up_feats3.append(pseudo_up_feat3)

                    flow_gts1 += utils.list_index_gather([gts[idx_pc]], [pseudo_fps_ind1]) if self.training else [None]
                    flow_gts2 += utils.list_index_gather([flow_gts1[idx_pc]], [pseudo_fps_ind2]) if self.training else [None]
                    flow_gts3 += utils.list_index_gather([flow_gts2[idx_pc]], [pseudo_fps_ind3]) if self.training else [None]
                    
            
            up_flows3.append(up_flow3)
            up_flows2.append(up_flow2)
            up_flows1.append(up_flow1)
            interp_flows.append(interp_flow)
            flow_preds3.append(flow_pred3)
            flow_preds2.append(flow_pred2)
            flow_preds1.append(flow_pred1)
            flow_preds0.append(flow_pred0)
            tot_loss_dif3 += loss_dif3
            tot_loss_dif2 += loss_dif2
            tot_loss_dif1 += loss_dif1
            tot_loss_dif0 += loss_dif0
        
        flow_preds = [flow_preds0, flow_preds1, flow_preds2, flow_preds3]
        fps_inds = [fps_inds1, fps_inds2, fps_inds3]
        tot_loss_difs = [tot_loss_dif0, tot_loss_dif1, tot_loss_dif2, tot_loss_dif3]
            
        return flow_preds, fps_inds, tot_loss_difs
    
class GenFlow_rec_seq_mask(nn.Module):
    def __init__(self, npoint):
        super(GenFlow_rec_seq_mask, self).__init__()
        self.npoint = npoint
        self.unit = GenFlow_unit_mask(self.npoint)
        
    def forward(self, pcs, feats, gts, masks):
        assert len(pcs) >= 2
        assert len(gts) >= len(pcs) - 1
        assert len(feats) == len(pcs)
        #print(len(pcs), len(gts))
        B = gts[0].size()[0]
        pcs_length = len(pcs)

        pcs = [pc.permute(0, 2, 1).contiguous() for pc in pcs]
        feats = [feat.permute(0, 2, 1).contiguous() for feat in feats]
        gts = [gt.permute(0, 2, 1).contiguous() for gt in gts] if self.training else [None for _ in range(len(gts))]
        masks = [mask.permute(0, 2, 1).contiguous() for mask in masks] if self.training else [None for _ in range(len(masks))]
        
        pcs_cat = torch.cat(pcs, dim=0) 
        feats_cat = torch.cat(feats, dim=0)
        
        ([fps_xyzs1, fps_xyzs2, fps_xyzs3], 
        [fps_inds1, fps_inds2, fps_inds3], 
        [up_feats0, up_feats1, up_feats2, up_feats3]) = self.unit.encode(pcs_cat, feats_cat)
        
        #fps_xyzs4 = torch.split(fps_xyzs4, B, dim=0)
        fps_xyzs3 = list(torch.split(fps_xyzs3, B, dim=0))
        fps_xyzs2 = list(torch.split(fps_xyzs2, B, dim=0))
        fps_xyzs1 = list(torch.split(fps_xyzs1, B, dim=0))

        fps_inds1 = list(torch.split(fps_inds1, B, dim=0))
        fps_inds2 = list(torch.split(fps_inds2, B, dim=0))
        fps_inds3 = list(torch.split(fps_inds3, B, dim=0))
        
        up_feats3 = list(torch.split(up_feats3, B, dim=0))
        up_feats2 = list(torch.split(up_feats2, B, dim=0))
        up_feats1 = list(torch.split(up_feats1, B, dim=0))
        up_feats0 = list(torch.split(up_feats0, B, dim=0))
        
        flow_gts1 = utils.list_index_gather(gts, fps_inds1) if self.training else [None for _ in range(len(pcs))]
        flow_gts2 = utils.list_index_gather(flow_gts1, fps_inds2) if self.training else [None for _ in range(len(pcs))]
        flow_gts3 = utils.list_index_gather(flow_gts2, fps_inds3) if self.training else [None for _ in range(len(pcs))]
        masks1 = utils.list_index_gather(masks, fps_inds1) if self.training else [None for _ in range(len(pcs))]
        masks2 = utils.list_index_gather(masks1, fps_inds2) if self.training else [None for _ in range(len(pcs))]
        flow_preds3, flow_preds2, flow_preds1, flow_preds0 = [], [], [], []
        up_flows3, up_flows2, up_flows1, interp_flows = [], [], [], []
        tot_loss_dif3, tot_loss_dif2, tot_loss_dif1, tot_loss_dif0 = 0, 0, 0, 0
        corr0, corr1, corr2, corr3 = None, None, None, None
        state0, state1, state2, state3 = None, None, None, None
        for idx_gt in range(len(gts)):
            idx_pc = idx_gt + 1
            
            if(self.training and idx_pc > 1 and idx_pc < pcs_length):
                pre_masks = [0, 1]
                mask_pre = pre_masks[np.random.randint(0, 2)] 
            else:
                mask_pre = 1

            future = (idx_pc >= pcs_length)
            ([corr0, corr1, corr2, corr3],
             [state0, state1, state2, state3],
             [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
             [up_flow1, up_flow2, up_flow3, interp_flow],
             [loss_dif0, loss_dif1, loss_dif2, loss_dif3]) = self.unit.decode([gts[idx_pc-1], flow_gts1[idx_pc-1], flow_gts2[idx_pc-1], flow_gts3[idx_pc-1]],
                                                                       [masks[idx_pc-1], masks1[idx_pc-1], masks2[idx_pc-1]],
                                                                       [pcs[idx_pc-2], fps_xyzs1[idx_pc-2], fps_xyzs2[idx_pc-2], fps_xyzs3[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       [pcs[idx_pc-1], fps_xyzs1[idx_pc-1], fps_xyzs2[idx_pc-1], fps_xyzs3[idx_pc-1]],
                                                                       [pcs[idx_pc], fps_xyzs1[idx_pc], fps_xyzs2[idx_pc], fps_xyzs3[idx_pc]] if(not future) else [None, None, None, None],
                                                                       [corr0, corr1, corr2, corr3], [state0, state1, state2, state3],
                                                                       [up_feats0[idx_pc-2], up_feats1[idx_pc-2], up_feats2[idx_pc-2], up_feats3[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       [up_feats0[idx_pc-1], up_feats1[idx_pc-1], up_feats2[idx_pc-1], up_feats3[idx_pc-1]],
                                                                       [up_feats0[idx_pc], up_feats1[idx_pc], up_feats2[idx_pc], up_feats3[idx_pc]] if(not future) else [None, None, None, None],
                                                                       [up_flows1[idx_pc-2], up_flows2[idx_pc-2], up_flows3[idx_pc-2], interp_flows[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       [flow_preds0[idx_pc-2], flow_preds1[idx_pc-2], flow_preds2[idx_pc-2], flow_preds3[idx_pc-2]] if(idx_pc > 1) else [None, None, None, None],
                                                                       mask_pre)
            if(future):
                corr0, corr1, corr2, corr3 = None, None, None, None
                if(idx_pc < len(gts)):
                    pseudo_pc = (pcs[idx_pc-1] + flow_pred0).detach()
                    pseudo_feat = (pcs[idx_pc-1] + flow_pred0).detach()
                    ([pseudo_fps_xyz1, pseudo_fps_xyz2, pseudo_fps_xyz3], 
                    [pseudo_fps_ind1, pseudo_fps_ind2, pseudo_fps_ind3], 
                    [pseudo_up_feat0, pseudo_up_feat1, pseudo_up_feat2, pseudo_up_feat3]) = self.unit.encode(pseudo_pc, pseudo_feat)

                    pcs.append(pseudo_pc)
                    fps_inds1.append(pseudo_fps_ind1)
                    fps_inds2.append(pseudo_fps_ind2)
                    fps_inds3.append(pseudo_fps_ind3)
                    fps_xyzs1.append(pseudo_fps_xyz1)
                    fps_xyzs2.append(pseudo_fps_xyz2)
                    fps_xyzs3.append(pseudo_fps_xyz3)
                    up_feats0.append(pseudo_up_feat0)
                    up_feats1.append(pseudo_up_feat1)
                    up_feats2.append(pseudo_up_feat2)
                    up_feats3.append(pseudo_up_feat3)

                    flow_gts1 += utils.list_index_gather([gts[idx_pc]], [pseudo_fps_ind1]) if self.training else [None]
                    flow_gts2 += utils.list_index_gather([flow_gts1[idx_pc]], [pseudo_fps_ind2]) if self.training else [None]
                    flow_gts3 += utils.list_index_gather([flow_gts2[idx_pc]], [pseudo_fps_ind3]) if self.training else [None]
                    masks1 += utils.list_index_gather([masks[idx_pc]], [pseudo_fps_ind1]) if self.training else [None]
                    masks2 += utils.list_index_gather([masks1[idx_pc]], [pseudo_fps_ind2]) if self.training else [None]
                    
            
            up_flows3.append(up_flow3)
            up_flows2.append(up_flow2)
            up_flows1.append(up_flow1)
            interp_flows.append(interp_flow)
            flow_preds3.append(flow_pred3)
            flow_preds2.append(flow_pred2)
            flow_preds1.append(flow_pred1)
            flow_preds0.append(flow_pred0)
            tot_loss_dif3 += loss_dif3
            tot_loss_dif2 += loss_dif2
            tot_loss_dif1 += loss_dif1
            tot_loss_dif0 += loss_dif0
        
        flow_preds = [flow_preds0, flow_preds1, flow_preds2, flow_preds3]
        fps_inds = [fps_inds1, fps_inds2, fps_inds3]
        tot_loss_difs = [tot_loss_dif0, tot_loss_dif1, tot_loss_dif2, tot_loss_dif3]
            
        return flow_preds, fps_inds, tot_loss_difs
    

def multiscalerecurrentloss_seq(gts, preds, fps_inds, weights, loss_difs, weight_dist, weight_dif):
    # gts: a list of tensors with shape [B, N, 3]
    # preds: a list of lists of tensors with shape [B, 3, N]
    # fps_inds: a list of lists of tensors with shape [B, N]

    dist_loss =  0
    dif_loss = 0
    for i in range(len(gts)): # time
        cur_gt = gts[i].permute(0, 2, 1).contiguous() #[B, 3, N]
        for j in range(len(preds)): # scale
            if(j > 0):
                cur_gt = pointnet2_utils.gather_operation(cur_gt, fps_inds[j-1][i])
            
            cur_pred = preds[j][i]
            cur_loss = torch.sum(torch.norm(cur_pred - cur_gt, dim=1), dim=1)
            #print('dist:', j, cur_loss.detach())
            dist_loss += weights[j] * cur_loss.mean()

    for k in range(len(loss_difs)):
        if(loss_difs[k] is not None):
            #print('dif:', k, loss_difs[k].detach())
            dif_loss += weights[k] * loss_difs[k].mean()

    tot_loss = weight_dist * dist_loss + weight_dif * dif_loss
            
    return tot_loss

def multiscalerecurrentloss_seq_mask(gts, masks, preds, fps_inds, weights, loss_difs, weight_dist, weight_dif):
    # gts: a list of tensors with shape [B, N, 3]
    # masks a list od tensors with shape [B, N, 1]
    # preds: a list of lists of tensors with shape [B, 3, N]
    # fps_inds: a list of lists of tensors with shape [B, N]

    dist_loss =  0
    dif_loss = 0
    for i in range(len(gts)): # time
        cur_gt = gts[i].permute(0, 2, 1).contiguous() #[B, 3, N]
        cur_mask = masks[i].permute(0, 2, 1).contiguous() #[B, 1, N]
        for j in range(len(preds)): # scale
            if(j > 0):
                cur_gt = pointnet2_utils.gather_operation(cur_gt, fps_inds[j-1][i])
                cur_mask = pointnet2_utils.gather_operation(cur_mask, fps_inds[j-1][i])
            
            cur_pred = preds[j][i]
            cur_loss = torch.norm(cur_pred - cur_gt, dim=1) * cur_mask.squeeze(1) #[B, N]
            cur_loss = torch.sum(cur_loss, dim=1) #[B]
            #print('dist:', j, cur_loss.detach())
            dist_loss += weights[j] * cur_loss.mean()

    for k in range(len(loss_difs)):
        if(loss_difs[k] is not None):
            #print('dif:', k, loss_difs[k].detach())
            dif_loss += weights[k] * loss_difs[k].mean()

    tot_loss = weight_dist * dist_loss + weight_dif * dif_loss
            
    return tot_loss

def multiscalerecurrentloss_dist(gts, preds, fps_inds, weights):
    # gt: a list of tensors with shape [B, N, 3]
    # pred: a list of lists of tensors with shape [B, 3, N]
    # fps_inds: a list of lists of tensors with shape [B, N]

    dist_loss =  0
    for i in range(len(gts)): # time
        cur_gt = gts[i].permute(0, 2, 1).contiguous() #[B, 3, N]
        for j in range(len(preds)): # scale
            if(j > 0):
                cur_gt = pointnet2_utils.gather_operation(cur_gt, fps_inds[j-1][i])
            
            cur_pred = preds[j][i]
            cur_loss = torch.sum(torch.norm(cur_pred - cur_gt, dim=1), dim=1)
            #print('dist:', j, cur_loss.detach())
            dist_loss += weights[j] * cur_loss.mean()

    tot_loss = dist_loss
            
    return tot_loss

def multiscalerecurrentloss_dif(loss_difs, weights):
    
    dif_loss = 0
    for k in range(len(loss_difs)):
        if(loss_difs[k] is not None):
            #print('dif:', k, loss_difs[k].detach())
            dif_loss += weights[k] * loss_difs[k].mean()
            
    tot_loss = dif_loss
    
    return tot_loss