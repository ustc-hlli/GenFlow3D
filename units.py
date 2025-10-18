# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 07:00:41 2024

@author: lihl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet2 import pointnet2_utils
import layers as layers
import utils

class Encoder(nn.Module):
    def __init__(self, npoint, dims=[32, 64, 128, 192, 192]):
        super(Encoder, self).__init__()
        self.npoint = npoint
        
        # encoder
        self.down0 = layers.Conv1d(3, dims[0])
        self.down1 = layers.SetConv(0.5, 2048, 16, dims[0], [dims[0], dims[1]])
        self.down2 = layers.SetConv(1.0, 512, 16, dims[1], [dims[1], dims[2]])
        self.down3 = layers.SetConv(2.0, 128, 16, dims[2], [dims[2], dims[3]])
        self.down4 = layers.SetConv(4.0, 64, 16, dims[3], [dims[3], dims[4]])
        
        self.up1 = layers.SetUpConv(0.75, 8, dims[1], dims[0], [dims[0]], [dims[0]])
        self.up2 = layers.SetUpConv(1.5, 8, dims[2], dims[1], [dims[1]], [dims[1]])
        self.up3 = layers.SetUpConv(3.0, 8, dims[3], dims[2], [dims[2]], [dims[2]])
        self.up4 = layers.SetUpConv(6.0, 8, dims[4], dims[3], [dims[3]], [dims[3]])
        
    def forward(self, pc, feat):
        if(pc is None):
            return ([None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None])
        down_feat0 = self.down0(feat)
        down_feat1, fps_xyz1, fps_ind1 = self.down1(pc, down_feat0)
        down_feat2, fps_xyz2, fps_ind2 = self.down2(fps_xyz1, down_feat1)
        down_feat3, fps_xyz3, fps_ind3 = self.down3(fps_xyz2, down_feat2)
        down_feat4, fps_xyz4, fps_ind4 = self.down4(fps_xyz3, down_feat3)
        
        up_feat3 = self.up4(fps_xyz4, fps_xyz3, down_feat4, down_feat3)
        up_feat2 = self.up3(fps_xyz3, fps_xyz2, up_feat3, down_feat2)
        up_feat1 = self.up2(fps_xyz2, fps_xyz1, up_feat2, down_feat1)
        up_feat0 = self.up1(fps_xyz1, pc, up_feat1, down_feat0)
        
        return ([fps_xyz1, fps_xyz2, fps_xyz3, fps_xyz4],
                [fps_ind1, fps_ind2, fps_ind3, fps_ind4],
                [up_feat0, up_feat1, up_feat2, up_feat3]) 
    
class GenFlow_unit(nn.Module):
    def __init__(self, npoint):
        super(GenFlow_unit, self).__init__()
        self.npoint = npoint
        
        # encoder
        self.encoder = Encoder(self.npoint)
                
        # rec_corr       
        self.hist0 = layers.SetUpdate_rec2_flow(self.npoint, 4, 32, 32)
        self.hist1 = layers.SetUpdate_rec2_flow(2048, 4, 64, 64)
        self.hist2 = layers.SetUpdate_rec2_flow(512, 4, 128, 128)
        self.hist3 = layers.SetUpdate_rec2_flow(128, 4, 192, 192)

        self.corr3 = layers.CorrLayer_bidcos(32, 192, 192, 0)
        self.corr2 = layers.CorrLayer_bidcos(32, 128, 128, 192)
        self.corr1 = layers.CorrLayer_bidcos(32, 64, 64, 128)
        self.corr0 = layers.CorrLayer_bidcos(32, 32, 32, 64)
        
        # decoder
        self.dif3 = layers.FlowEstimationLayer_flow(3.0, 128, 8, 192*3, 192)
        self.dif2 = layers.DiffusionLayer_rec2_transformer(1.5, 512, 4, 8, 12, 128*3, 192, 128)
        self.dif1 = layers.DiffusionLayer_rec2_transformer(0.75, 2048, 4, 8, 12, 64*3, 128, 64)
        self.dif0 = layers.DiffusionLayer_rec2_transformer(0.75, self.npoint, 4, 8, 12, 32*3, 64, 32)
        
        self.warper = utils.PointWarping()
        self.upsampler = utils.UpsampleFlow()

    def encode(self, pc, feat):
        ([fps_xyz1, fps_xyz2, fps_xyz3, _], 
         [fps_indx1, fps_indx2, fps_indx3, _], 
         [up_feat0, up_feat1, up_feat2, up_feat3]) = self.encoder(pc, feat)
        
        return ([fps_xyz1, fps_xyz2, fps_xyz3], 
         [fps_indx1, fps_indx2, fps_indx3], 
         [up_feat0, up_feat1, up_feat2, up_feat3])
    
    def decode(self, gts, pp_xyzs, p_xyzs, xyzs, corrs, states, pp_feats, p_feats, feats, p_raw_flows, p_flows, mask_pre=True):
        for fi in range(len(p_raw_flows)):
            if(p_raw_flows[fi] is not None):
                p_raw_flows[fi] = p_raw_flows[fi].detach()
        for fi in range(len(p_flows)):
            if(p_flows[fi] is not None):
                p_flows[fi] = p_flows[fi].detach()            
        if(p_flows[0] is not None):
            flow = p_flows[0].detach()
            warped_p_xyz = self.warper(pp_xyzs[0], p_xyzs[3], flow)
            interp_flow = self.upsampler(warped_p_xyz, pp_xyzs[0], flow)
        else:
            interp_flow = None
            
        if(mask_pre):
            masked_interp_flow = interp_flow
        else:
            masked_interp_flow = None
        
        corr3 = self.corr3(p_xyzs[3], xyzs[3], None, p_feats[3], feats[3], None, masked_interp_flow)
        if(corrs[3] is None and pp_xyzs[3] is not None): # future pseudo point clouds
            corrs[3] = self.corr3(pp_xyzs[3], p_xyzs[3], None, pp_feats[3], p_feats[3], None, p_raw_flows[3])
        state3 = self.hist3(pp_xyzs[3],
                                p_xyzs[3],
                                states[3], corrs[3],
                                pp_feats[3],
                                p_feats[3], p_flows[3])

        masked_corr3 = corr3
        masked_state3 = state3 * mask_pre
        condition3 = torch.cat([masked_state3, masked_corr3, p_feats[3]], dim=1)
        geo3, flow_pred3 = self.dif3(p_xyzs[3], condition3, masked_interp_flow) 
        loss_dif3 = torch.zeros([condition3.size()[0]]).to(condition3.device)       
        up_flow_pred3 = self.upsampler(p_xyzs[2], p_xyzs[3], flow_pred3)
        
        corr2 = self.corr2(p_xyzs[2], xyzs[2], p_xyzs[3], p_feats[2], feats[2], corr3, up_flow_pred3)
        if(corrs[2] is None and pp_xyzs[2] is not None):
            corrs[2] = self.corr2(pp_xyzs[2], p_xyzs[2], pp_xyzs[3], pp_feats[2], p_feats[2], corrs[3], p_raw_flows[2])
        state2 = self.hist2(pp_xyzs[2],
                                p_xyzs[2],
                                states[2], corrs[2],
                                pp_feats[2],
                                p_feats[2], p_flows[2])
        
        masked_corr2 = corr2
        masked_state2 = state2 * mask_pre
        condition2 = torch.cat([masked_state2, masked_corr2, p_feats[2]], dim=1)
        geo2, flow_pred2, loss_dif2 = self.dif2(p_xyzs[2], p_xyzs[3],
                                               condition2, geo3, 
                                               up_flow_pred3, gts[2])
        up_flow_pred2 = self.upsampler(p_xyzs[1], p_xyzs[2], flow_pred2)
        
        corr1 = self.corr1(p_xyzs[1], xyzs[1], p_xyzs[2], p_feats[1], feats[1], corr2, up_flow_pred2)
        if(corrs[1] is None and pp_xyzs[1] is not None):
            corrs[1] = self.corr1(pp_xyzs[1], p_xyzs[1], pp_xyzs[2], pp_feats[1], p_feats[1], corrs[2], p_raw_flows[1])
        state1 = self.hist1(pp_xyzs[1],
                                p_xyzs[1],
                                states[1], corrs[1],
                                pp_feats[1],
                                p_feats[1], p_flows[1])

        masked_corr1 = corr1
        masked_state1 = state1 * mask_pre
        condition1 = torch.cat([masked_state1, masked_corr1, p_feats[1]], dim=1)
        geo1, flow_pred1, loss_dif1 = self.dif1(p_xyzs[1], p_xyzs[2],
                                               condition1, geo2, 
                                               up_flow_pred2, gts[1])
        up_flow_pred1 = self.upsampler(p_xyzs[0], p_xyzs[1], flow_pred1)
        
        corr0 = self.corr0(p_xyzs[0], xyzs[0], p_xyzs[1], p_feats[0], feats[0], corr1, up_flow_pred1)
        if(corrs[0] is None and pp_xyzs[0] is not None):
            corrs[0] = self.corr0(pp_xyzs[0], p_xyzs[0], pp_xyzs[1], pp_feats[0], p_feats[0], corrs[1], p_raw_flows[0])
        state0 = self.hist0(pp_xyzs[0],
                                p_xyzs[0],
                                states[0], corrs[0],
                                pp_feats[0],
                                p_feats[0], p_flows[0])

        masked_corr0 = corr0
        masked_state0 = state0 * mask_pre
        condition0 = torch.cat([masked_state0, masked_corr0, p_feats[0]], dim=1)
        geo0, flow_pred0, loss_dif0 = self.dif0(p_xyzs[0], p_xyzs[1],
                                               condition0, geo1, 
                                               up_flow_pred1, gts[0])
        
        
        return ([corr0, corr1, corr2, corr3],
        [state0, state1, state2, state3],
        [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
        [up_flow_pred1, up_flow_pred2, up_flow_pred3, interp_flow],
        [loss_dif0, loss_dif1, loss_dif2, loss_dif3])
        
    def forward(self, pc, feat, gts, pp_xyzs, p_xyzs, corrs, states, pp_feats, p_feats, p_raw_flows, p_flows, mask_pre, mask_fut, pseudo_feat=True):
        # not used
        for fi in range(len(p_raw_flows)):
            if(p_raw_flows[fi] is not None):
                p_raw_flows[fi] = p_raw_flows[fi].detach()
        for fi in range(len(p_flows)):
            if(p_flows[fi] is not None):
                p_flows[fi] = p_flows[fi].detach()

        ([fps_xyz1, fps_xyz2, fps_xyz3], 
         [fps_indx1, fps_indx2, fps_indx3], 
         [up_feat0, up_feat1, up_feat2, up_feat3]) = self.encode(pc, feat)
        
        if(gts[0] is None): # the first point cloud           
            return ([fps_xyz1, fps_xyz2, fps_xyz3],
                    [fps_indx1, fps_indx2, fps_indx3], 
                    [up_feat0, up_feat1, up_feat2, up_feat3],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None],
                    [0, 0, 0, 0])
        
             
        ([corr0, corr1, corr2, corr3],
        [state0, state1, state2, state3],
        [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
        [up_flow_pred1, up_flow_pred2, up_flow_pred3, interp_flow],
        [loss_dif0, loss_dif1, loss_dif2, loss_dif3]) = self.decode(gts, pp_xyzs, p_xyzs, [pc, fps_xyz1, fps_xyz2, fps_xyz3],
                                                                    corrs, states, pp_feats, p_feats, [up_feat0, up_feat1, up_feat2, up_feat3],
                                                                    p_raw_flows, p_flows, mask_pre)
        
        
        if(pc is None and pseudo_feat):
            pc = (p_xyzs[0] + flow_pred0).detach()
            feat = (p_xyzs[0] + flow_pred0).detach()
            ([fps_xyz1, fps_xyz2, fps_xyz3, _], 
             [fps_indx1, fps_indx2, fps_indx3, _], 
             [up_feat0, up_feat1, up_feat2, up_feat3]) = self.encode(pc, feat)
            
        return ([fps_xyz1, fps_xyz2, fps_xyz3],
                [fps_indx1, fps_indx2, fps_indx3], 
                [up_feat0, up_feat1, up_feat2, up_feat3],
                [corr0, corr1, corr2, corr3],
                [state0, state1, state2, state3],
                [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
                [up_flow_pred1, up_flow_pred2, up_flow_pred3, interp_flow],
                [loss_dif0, loss_dif1, loss_dif2, loss_dif3])
    
class GenFlow_unit_mask(nn.Module):
    def __init__(self, npoint):
        super(GenFlow_unit_mask, self).__init__()
        self.npoint = npoint
        
        # encoder
        self.encoder = Encoder(self.npoint)
                
        # rec_corr       
        self.hist0 = layers.SetUpdate_rec2_flow(self.npoint, 4, 32, 32)
        self.hist1 = layers.SetUpdate_rec2_flow(2048, 4, 64, 64)
        self.hist2 = layers.SetUpdate_rec2_flow(512, 4, 128, 128)
        self.hist3 = layers.SetUpdate_rec2_flow(128, 4, 192, 192)

        self.corr3 = layers.CorrLayer_bidcos(32, 192, 192, 0)
        self.corr2 = layers.CorrLayer_bidcos(32, 128, 128, 192)
        self.corr1 = layers.CorrLayer_bidcos(32, 64, 64, 128)
        self.corr0 = layers.CorrLayer_bidcos(32, 32, 32, 64)
        
        # decoder
        self.dif3 = layers.FlowEstimationLayer_flow(3.0, 128, 8, 192*3, 192)
        self.dif2 = layers.DiffusionLayer_rec2_transformer_mask(1.5, 512, 4, 8, 12, 128*3, 192, 128)
        self.dif1 = layers.DiffusionLayer_rec2_transformer_mask(0.75, 2048, 4, 8, 12, 64*3, 128, 64)
        self.dif0 = layers.DiffusionLayer_rec2_transformer_mask(0.75, self.npoint, 4, 8, 12, 32*3, 64, 32)

        self.warper = utils.PointWarping()
        self.upsampler = utils.UpsampleFlow()

    def encode(self, pc, feat):
        ([fps_xyz1, fps_xyz2, fps_xyz3, _], 
         [fps_indx1, fps_indx2, fps_indx3, _], 
         [up_feat0, up_feat1, up_feat2, up_feat3]) = self.encoder(pc, feat)
        
        return ([fps_xyz1, fps_xyz2, fps_xyz3], 
         [fps_indx1, fps_indx2, fps_indx3], 
         [up_feat0, up_feat1, up_feat2, up_feat3])
    
    def decode(self, gts, valid_masks, pp_xyzs, p_xyzs, xyzs, corrs, states, pp_feats, p_feats, feats, p_raw_flows, p_flows, mask_pre=True):
        for fi in range(len(p_raw_flows)):
            if(p_raw_flows[fi] is not None):
                p_raw_flows[fi] = p_raw_flows[fi].detach()
        for fi in range(len(p_flows)):
            if(p_flows[fi] is not None):
                p_flows[fi] = p_flows[fi].detach()            
        if(p_flows[0] is not None):
            flow = p_flows[0].detach()
            warped_p_xyz = self.warper(pp_xyzs[0], p_xyzs[3], flow)
            interp_flow = self.upsampler(warped_p_xyz, pp_xyzs[0], flow)
        else:
            interp_flow = None
        
        if(mask_pre):
            masked_interp_flow = interp_flow
        else:
            masked_interp_flow = None

        corr3 = self.corr3(p_xyzs[3], xyzs[3], None, p_feats[3], feats[3], None, masked_interp_flow)
        if(corrs[3] is None and pp_xyzs[3] is not None): # future pseudo point clouds
            corrs[3] = self.corr3(pp_xyzs[3], p_xyzs[3], None, pp_feats[3], p_feats[3], None, p_raw_flows[3])
        state3 = self.hist3(pp_xyzs[3],
                                p_xyzs[3],
                                states[3], corrs[3],
                                pp_feats[3],
                                p_feats[3], p_flows[3])

        masked_corr3 = corr3
        masked_state3 = state3 * mask_pre
        condition3 = torch.cat([masked_state3, masked_corr3, p_feats[3]], dim=1)
        geo3, flow_pred3 = self.dif3(p_xyzs[3], condition3, masked_interp_flow) 
        loss_dif3 = torch.zeros([condition3.size()[0]]).to(condition3.device)       
        up_flow_pred3 = self.upsampler(p_xyzs[2], p_xyzs[3], flow_pred3)
        
        corr2 = self.corr2(p_xyzs[2], xyzs[2], p_xyzs[3], p_feats[2], feats[2], corr3, up_flow_pred3)
        if(corrs[2] is None and pp_xyzs[2] is not None):
            corrs[2] = self.corr2(pp_xyzs[2], p_xyzs[2], pp_xyzs[3], pp_feats[2], p_feats[2], corrs[3], p_raw_flows[2])
        state2 = self.hist2(pp_xyzs[2],
                                p_xyzs[2],
                                states[2], corrs[2],
                                pp_feats[2],
                                p_feats[2], p_flows[2])

        masked_corr2 = corr2
        masked_state2 = state2 * mask_pre
        condition2 = torch.cat([masked_state2, masked_corr2, p_feats[2]], dim=1)
        geo2, flow_pred2, loss_dif2 = self.dif2(p_xyzs[2], p_xyzs[3],
                                               condition2, geo3, 
                                               up_flow_pred3, gts[2], valid_masks[2])
        up_flow_pred2 = self.upsampler(p_xyzs[1], p_xyzs[2], flow_pred2)
        
        corr1 = self.corr1(p_xyzs[1], xyzs[1], p_xyzs[2], p_feats[1], feats[1], corr2, up_flow_pred2)
        if(corrs[1] is None and pp_xyzs[1] is not None):
            corrs[1] = self.corr1(pp_xyzs[1], p_xyzs[1], pp_xyzs[2], pp_feats[1], p_feats[1], corrs[2], p_raw_flows[1])
        state1 = self.hist1(pp_xyzs[1],
                                p_xyzs[1],
                                states[1], corrs[1],
                                pp_feats[1],
                                p_feats[1], p_flows[1])
        
        masked_corr1 = corr1
        masked_state1 = state1 * mask_pre
        condition1 = torch.cat([masked_state1, masked_corr1, p_feats[1]], dim=1)
        geo1, flow_pred1, loss_dif1 = self.dif1(p_xyzs[1], p_xyzs[2],
                                               condition1, geo2, 
                                               up_flow_pred2, gts[1], valid_masks[1])
        up_flow_pred1 = self.upsampler(p_xyzs[0], p_xyzs[1], flow_pred1)
        
        corr0 = self.corr0(p_xyzs[0], xyzs[0], p_xyzs[1], p_feats[0], feats[0], corr1, up_flow_pred1)
        if(corrs[0] is None and pp_xyzs[0] is not None):
            corrs[0] = self.corr0(pp_xyzs[0], p_xyzs[0], pp_xyzs[1], pp_feats[0], p_feats[0], corrs[1], p_raw_flows[0])
        state0 = self.hist0(pp_xyzs[0],
                                p_xyzs[0],
                                states[0], corrs[0],
                                pp_feats[0],
                                p_feats[0], p_flows[0])

        masked_corr0 = corr0
        masked_state0 = state0 * mask_pre
        condition0 = torch.cat([masked_state0, masked_corr0, p_feats[0]], dim=1)
        geo0, flow_pred0, loss_dif0 = self.dif0(p_xyzs[0], p_xyzs[1],
                                               condition0, geo1, 
                                               up_flow_pred1, gts[0], valid_masks[0])
        
        
        return ([corr0, corr1, corr2, corr3],
        [state0, state1, state2, state3],
        [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
        [up_flow_pred1, up_flow_pred2, up_flow_pred3, interp_flow],
        [loss_dif0, loss_dif1, loss_dif2, loss_dif3])
        
    def forward(self, pc, feat, gts, valids, pp_xyzs, p_xyzs, corrs, states, pp_feats, p_feats, p_raw_flows, p_flows, mask_pre, pseudo_feat=True):
        # not used
        for fi in range(len(p_raw_flows)):
            if(p_raw_flows[fi] is not None):
                p_raw_flows[fi] = p_raw_flows[fi].detach()
        for fi in range(len(p_flows)):
            if(p_flows[fi] is not None):
                p_flows[fi] = p_flows[fi].detach()

        ([fps_xyz1, fps_xyz2, fps_xyz3], 
         [fps_indx1, fps_indx2, fps_indx3], 
         [up_feat0, up_feat1, up_feat2, up_feat3]) = self.encode(pc, feat)
        
        if(gts[0] is None): # the first point cloud           
            return ([fps_xyz1, fps_xyz2, fps_xyz3],
                    [fps_indx1, fps_indx2, fps_indx3], 
                    [up_feat0, up_feat1, up_feat2, up_feat3],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None],
                    [0, 0, 0, 0])
        
             
        ([corr0, corr1, corr2, corr3],
        [state0, state1, state2, state3],
        [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
        [up_flow_pred1, up_flow_pred2, up_flow_pred3, interp_flow],
        [loss_dif0, loss_dif1, loss_dif2, loss_dif3]) = self.decode(gts, valids, pp_xyzs, p_xyzs, [pc, fps_xyz1, fps_xyz2, fps_xyz3],
                                                                    corrs, states, pp_feats, p_feats, [up_feat0, up_feat1, up_feat2, up_feat3],
                                                                    p_raw_flows, p_flows, mask_pre)
        
        
        if(pc is None and pseudo_feat):
            pc = (p_xyzs[0] + flow_pred0).detach()
            feat = (p_xyzs[0] + flow_pred0).detach()
            ([fps_xyz1, fps_xyz2, fps_xyz3, _], 
             [fps_indx1, fps_indx2, fps_indx3, _], 
             [up_feat0, up_feat1, up_feat2, up_feat3]) = self.encode(pc, feat)
            
        return ([fps_xyz1, fps_xyz2, fps_xyz3],
                [fps_indx1, fps_indx2, fps_indx3], 
                [up_feat0, up_feat1, up_feat2, up_feat3],
                [corr0, corr1, corr2, corr3],
                [state0, state1, state2, state3],
                [flow_pred0, flow_pred1, flow_pred2, flow_pred3],
                [up_flow_pred1, up_flow_pred2, up_flow_pred3, interp_flow],
                [loss_dif0, loss_dif1, loss_dif2, loss_dif3])