# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pointnet2 import pointnet2_utils
import diffusion_utils
import utils

USE_GN = True
USE_BN = False

class Conv1d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1, padding=0, groups=1, bias=True, use_gn=USE_GN, use_bn=USE_BN, use_act=True):
        super(Conv1d, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        if(use_act):
            if(use_gn):
                self.composed_module = nn.Sequential(
            			nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
                        nn.GroupNorm(out_feat//16, out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
            elif(use_bn):
                self.composed_module = nn.Sequential(
            			nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
                        nn.BatchNorm1d(out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
            else:
                self.composed_module = nn.Sequential(
            			nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
        else:
            self.composed_module = nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, feat):
        feat = self.composed_module(feat)
        return feat
    
class Conv2d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1, padding=0, groups=1, bias=True, use_gn=USE_GN, use_bn=USE_BN, use_act=True):
        super(Conv2d, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        if(use_act):
            if(use_gn):
                self.composed_module = nn.Sequential(
            			nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
                        nn.GroupNorm(out_feat//16, out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
            elif(use_bn):
                self.composed_module = nn.Sequential(
            			nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
                        nn.BatchNorm2d(out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
            else:
                self.composed_module = nn.Sequential(
            			nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
        else:
            self.composed_module = nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, feat):
        feat = self.composed_module(feat)
        return feat
    
    
class SetConv(nn.Module):
    def __init__(self, radius, npoint, nsample, in_feat, mlp=[], last_act=True):
        super(SetConv, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.knn = radius == 0
        
        self.mlp = nn.ModuleList()
        last_feat = in_feat + 3
        for i in range(len(mlp)):
            if(i < len(mlp)-1 or last_act):
                self.mlp.append(Conv2d(last_feat, mlp[i]))
            else:
                self.mlp.append(Conv2d(last_feat, mlp[i], use_act=False))
            last_feat = mlp[i]
            
    def forward(self, xyz, feat):
        
        B, C, N = feat.size()
        
        xyz_t = xyz.permute(0 ,2, 1).contiguous()
        feat_t = feat.permute(0 ,2, 1).contiguous()
        
        if(N > self.npoint):
            fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint) #[B, N']
            new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx) #[B, 3, N']
        else:
            fps_idx = None
            new_xyz = xyz
            
        if(self.knn):
            _, group_idx = pointnet2_utils.knn(self.nsample, new_xyz.permute(0 ,2, 1).contiguous(), xyz_t) #[B, N', S]
        else:
            group_idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz_t, new_xyz.permute(0 ,2, 1).contiguous())
           
        group_feat = pointnet2_utils.grouping_operation(feat, group_idx) #[B, C, N', S]
        group_xyz = pointnet2_utils.grouping_operation(xyz, group_idx) - new_xyz.unsqueeze(3)
        group_feat = torch.cat([group_feat, group_xyz], dim=1)
        
        for layer in self.mlp:
            group_feat = layer(group_feat)
            
        new_feat = torch.max(group_feat, dim=3)[0]
        
        return new_feat, new_xyz, fps_idx
    
class SetUpConv(nn.Module):
    def __init__(self, radius, nsample, in_feat1, in_feat2, mlp1=[], mlp2=[], last_act=True):
        super(SetUpConv, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = radius == 0
        self.mlp1 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()

        last_feat = in_feat1 + 3
        for i in range(len(mlp1)):
            if(len(mlp2) == 0 and i == len(mlp1)-1 and (not last_act)):
                self.mlp1.append(Conv2d(last_feat, mlp1[i], use_act=False))
            else:
                self.mlp1.append(Conv2d(last_feat, mlp1[i]))
            last_feat = mlp1[i]
        last_feat = last_feat + in_feat2
        for j in range(len(mlp2)):
            if(j < len(mlp2)-1 or last_act):
                self.mlp2.append(Conv1d(last_feat, mlp2[j]))
            else:
                self.mlp2.append(Conv1d(last_feat, mlp2[j], use_act=False))
            last_feat = mlp2[j]

    def forward(self, xyz1, xyz2, feat1, feat2):
        '''
        xyz2 is the queries and xyz1 is the targets
        '''
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()

        if(self.knn):
            _, idx = pointnet2_utils.knn(self.nsample, xyz2_t, xyz1_t)
        else:
            idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz1_t, xyz2_t)

        group_feat = pointnet2_utils.grouping_operation(feat1, idx) #[B, C1, N2, S]
        group_xyz = pointnet2_utils.grouping_operation(xyz1, idx) - xyz2.unsqueeze(3) #[B, 3, N2, S]
        new_feat = torch.cat([group_feat, group_xyz], dim=1)

        for layer in self.mlp1:
            new_feat = layer(new_feat)
        new_feat = torch.max(new_feat, dim=3)[0] #[B, C', N2]

        if(feat2 is not None):
            new_feat = torch.cat([feat2, new_feat], dim=1)
        for layer in self.mlp2:
            new_feat = layer(new_feat)
        
        return new_feat
    
class SetPropagation(nn.Module):
    def __init__(self, in_feat1, in_feat2, mlp=[], last_act=True):
        super(SetPropagation, self).__init__()

        self.mlp = nn.ModuleList()
        last_feat = in_feat1 + in_feat2
        for i in range(len(mlp)):
            if(i < len(mlp)-1 or last_act):
                self.mlp.append(Conv1d(last_feat, mlp[i]))
            else:
                self.mlp.append(Conv1d(last_feat, mlp[i], use_act=False))
            last_feat = mlp[i]


    def forward(self, xyz1, xyz2, feat1, feat2):
        '''
        xyz2 is the queries and xyz1 is the targets
        '''
        
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()

        dists, idx = pointnet2_utils.knn(3, xyz2_t, xyz1_t) #[B, N2, S]
        group_feat1 = pointnet2_utils.grouping_operation(feat1, idx) #[B, C1, N2, S]
        weights = 1.0/(dists + 1e-8)
        weights = weights / (weights.sum(dim=2, keepdim=True))

        interp_feat1 = (group_feat1 * weights.unsqueeze(1)).sum(dim=3) #[B, C1, N2]
        if(feat2 is not None):
            new_feat = torch.cat([feat2, interp_feat1], dim=1)
        else:
            new_feat = interp_feat1

        for layer in self.mlp:
            new_feat = layer(new_feat)

        return new_feat
    
class FlowEmbedding(nn.Module):
    def __init__(self, in_feat1, in_feat2, k, mlp=[], last_act=True):
        super(FlowEmbedding, self).__init__()
        self.in_feat1 = in_feat1
        self.in_feat2 = in_feat2
        self.k = k

        self.mlp = nn.ModuleList()
        last_feat = in_feat1 + in_feat2 + 3
        for i in range(len(mlp)):
            if(i < len(mlp)-1 or last_act):
                self.mlp.append(Conv2d(last_feat, mlp[i]))
            else:
                self.mlp.append(Conv2d(last_feat, mlp[i], use_act=False))
            last_feat = mlp[i]
    
    def forward(self, xyz1, xyz2, feat1, feat2):
        B, C, N = feat1.size()
        if(xyz2 is None):
            xyz2 = torch.zeros_like(xyz1).to(xyz1.device)
            feat2 = torch.zeros_like(feat1).to(feat1.device)

        dists = torch.norm(xyz1.unsqueeze(3) - xyz2.unsqueeze(2), dim=1)
        _, k_idx = torch.topk(dists, self.k, dim=2, largest=False, sorted=False)

        group_xyz = pointnet2_utils.grouping_operation(xyz2, k_idx.int()) - xyz1.unsqueeze(3)
        group_feat = pointnet2_utils.grouping_operation(feat2, k_idx.int())

        new_feat = torch.cat([feat1.unsqueeze(3).repeat(1, 1, 1, self.k), group_feat, group_xyz], dim=1)
        for layer in self.mlp:
            new_feat = layer(new_feat)

        new_feat = torch.max(new_feat, dim=3)[0]

        return new_feat

class CrossLayerLight(nn.Module):
    def __init__(self, nsample, in_channel, mlp1, mlp2, use_gn=USE_GN, use_bn=USE_BN):
        super(CrossLayerLight,self).__init__()

        self.nsample = nsample
        self.use_gn = use_gn
        self.use_bn = use_bn
        self.pos1 = nn.Conv2d(3, mlp1[0], 1)
        self.mlp1 = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp1[0], 1)
        self.bias1 = nn.Parameter(torch.randn((1, mlp1[0], 1, 1)),requires_grad=True)
        if(use_gn):
            self.bn1 = nn.GroupNorm(mlp1[0]//16, mlp1[0])
        elif(use_bn):
            self.bn1 = nn.BatchNorm2d(mlp1[0])
        else:
            self.bn1 = nn.Identity()

        for i in range(1, len(mlp1)):
            self.mlp1.append(Conv2d(mlp1[i-1], mlp1[i], use_gn=self.use_gn))
            last_channel = mlp1[i]
        
        self.cross_t1 = nn.Conv1d(mlp1[-1], mlp2[0], 1)
        self.cross_t2 = nn.Conv1d(mlp1[-1], mlp2[0], 1)

        self.pos2 = nn.Conv2d(3, mlp2[0], 1)
        self.bias2 = nn.Parameter(torch.randn((1, mlp2[0], 1, 1)),requires_grad=True)
        if(use_gn):
            self.bn2 = nn.GroupNorm(mlp2[0]//16, mlp1[0])
        elif(use_bn):
            self.bn2 = nn.BatchNorm2d(mlp2[0])
        else:
            self.bn2 = nn.Identity()

        self.mlp2 = nn.ModuleList()
        for i in range(1, len(mlp2)):
            self.mlp2.append(Conv2d(mlp2[i-1], mlp2[i], use_gn=self.use_gn))
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, pos, mlp, bn):
        B, _, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, C1, _ = points1.shape
        _, C2, _ = points2.shape

        dists = torch.norm(xyz1.unsqueeze(3) - xyz2.unsqueeze(2), dim=1)
        _, knn_idx = torch.topk(dists, self.nsample, dim=2, largest=False, sorted=False)        
        neighbor_xyz = pointnet2_utils.grouping_operation(xyz2, knn_idx.int()) #[B, 3, N1, S]
        direction_xyz = neighbor_xyz - xyz1.view(B, 3, N1, 1)
        grouped_points2 = pointnet2_utils.grouping_operation(points2, knn_idx.int()) #[B, C2, N1, S]
        grouped_points1 = points1.view(B, C1, N1, 1).repeat(1, 1, 1, self.nsample)

        direction_xyz = pos(direction_xyz)
        new_points = self.relu(bn(grouped_points2 + grouped_points1 + direction_xyz))

        for i, conv in enumerate(mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (1, self.nsample)).squeeze(3)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2):
        B, _, N1 = pc1.size()
        
        if(pc2 is not None):
            feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
            feat1_new = self.cross_t1(feat1_new)
            feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), self.pos1, self.mlp1, self.bn1)
            feat2_new = self.cross_t2(feat2_new)

            feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)

            return feat1_new, feat2_new, feat1_final
        else:
            pc2 = torch.zeros_like(pc1).to(pc1.device)
            feat2 = torch.zeros_like(feat1).to(feat1.device)
            
            feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), self.pos1, self.mlp1, self.bn1)
            feat1_new = self.cross_t1(feat1_new)
            feat2_new = torch.zeros_like(feat1_new).to(feat1_new.device)
            
            feat1_final = self.cross(pc1, pc2, feat1_new, feat2_new, self.pos2, self.mlp2, self.bn2)
            
            return feat1_new, None, feat1_final

class BidirectionalLayerFeatCosine(nn.Module):
    def __init__(self, nsample, in_channel, mlp, use_gn=USE_GN, use_bn=USE_BN):
        super(BidirectionalLayerFeatCosine,self).__init__()

        self.nsample = nsample
        self.use_gn = use_gn
        self.use_bn = use_bn
        self.pos = nn.Conv2d(3, mlp[0], 1)
        self.mlp = nn.ModuleList()
        last_channel = in_channel
        self.cross_t11 = nn.Conv1d(in_channel, mlp[0], 1)
        self.cross_t22 = nn.Conv1d(in_channel, mlp[0], 1)
        self.bias = nn.Parameter(torch.randn((1, mlp[0], 1, 1)),requires_grad=True)
        if(use_gn):
            self.bn = nn.GroupNorm(mlp[0]//16, mlp[0])
        elif(use_bn):
            self.bn = nn.BatchNorm2d(mlp[0])
        else:
            self.bn = nn.Identity()

        for i in range(1, len(mlp)):
            self.mlp.append(Conv2d(mlp[i-1], mlp[i], use_gn = self.use_gn, use_bn = self.use_bn))

        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def cross(self, xyz1, xyz2, points1, points2, knn1, knn2, nsample=None):
        B, _, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, C1, _ = points1.shape
        _, C2, _ = points2.shape


        if nsample is None:
            nsample=self.nsample

        feat_dists = utils.cosine_distance(knn1.permute(0, 2, 1), knn2.permute(0, 2, 1))
        point_dists = torch.norm(xyz1.unsqueeze(3) - xyz2.unsqueeze(2), dim=1)
        _, knn_idx_feat = torch.topk(feat_dists, nsample//2, dim=2, largest=False, sorted=False)
        _, knn_idx_point = torch.topk(point_dists, nsample//2, dim=2, largest=False, sorted=False)
        knn_idx = torch.cat([knn_idx_point, knn_idx_feat], dim=2) #[B, N, K]
        
        neighbor_xyz = pointnet2_utils.grouping_operation(xyz2, knn_idx.int())
        direction_xyz = neighbor_xyz - xyz1.view(B, 3, N1, 1)

        grouped_points2 = pointnet2_utils.grouping_operation(points2, knn_idx.int()) #[B, C2, N1, S]
        grouped_points1 = points1.view(B, C1, N1, 1).repeat(1, 1, 1, nsample)

        direction_xyz = self.pos(direction_xyz)
        new_points = self.relu(self.bn(grouped_points2 + grouped_points1 + direction_xyz)) #[B, C, N1, S]

        for i, conv in enumerate(self.mlp):
            new_points = conv(new_points)
        
        new_points = F.max_pool2d(new_points, (1, nsample)).squeeze(3)

        return new_points

    def forward(self, pc1, pc2, feat1, feat2, knn1, knn2):
        if(pc2 is None):
            pc2 = torch.zeros_like(pc1).to(pc1.device)
            feat2 = torch.zeros_like(feat1).to(feat1.device)
            knn2 = torch.zeros_like(knn1).to(knn1.device)
            
            feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), knn1, knn2)
            
            return feat1_new, None
        else:
            feat1_new = self.cross(pc1, pc2, self.cross_t11(feat1),  self.cross_t22(feat2), knn1, knn2)

            feat2_new = self.cross(pc2, pc1, self.cross_t11(feat2), self.cross_t22(feat1), knn2, knn1)
            
            return feat1_new, feat2_new
    
class CorrLayer_bidcos(nn.Module):
    def __init__(self, ncorr, in_feat, corr_feat, down_feat):
        super(CorrLayer_bidcos, self).__init__()
        self.ncorr = ncorr
        self.in_feat = corr_feat
        self.corr_feat = corr_feat
        self.down_feat = down_feat

        self.warper = utils.PointWarping()
        self.bid_layer = BidirectionalLayerFeatCosine(self.ncorr, in_feat, [corr_feat, corr_feat])
        self.corr_layer = FlowEmbedding(corr_feat, corr_feat, self.ncorr, [corr_feat, corr_feat])
        if(down_feat > 0):
            self.up_layer = SetUpConv(0, 8, down_feat, corr_feat, [corr_feat], [corr_feat])
        
    def forward(self, xyz1, xyz2, xyz1d, feat1, feat2, corrd, flow):
        
        B, _, N = xyz1.size()
        if(xyz2 is not None):
            warped_xyz2 = self.warper(xyz1, xyz2, flow)
        else:
            warped_xyz2 = None
        feat1_new, feat2_new = self.bid_layer(xyz1, warped_xyz2, feat1, feat2, feat1, feat2) 
        corr_feat = self.corr_layer(xyz1, warped_xyz2, feat1_new, feat2_new)   
        if(xyz1d is not None):
            corr_feat = self.up_layer(xyz1d, xyz1, corrd, corr_feat)

        return corr_feat
    
class PointTransformer(nn.Module):
    def __init__(self, nsample, in_feat, out_feat):
        super(PointTransformer, self).__init__()   
        self.nsample = nsample
        
        self.pre_fc = nn.Conv1d(in_feat, out_feat, kernel_size=1)
        self.pos_fc = nn.Sequential(Conv2d(3, out_feat),
                                    nn.Conv2d(out_feat, out_feat, kernel_size=1))
        self.post_fc = Conv1d(out_feat, out_feat)
        self.att_fc = nn.Sequential(Conv2d(out_feat, out_feat),
                                    nn.Conv2d(out_feat, out_feat, kernel_size=1))
        
        self.wq = nn.Conv1d(out_feat, out_feat, kernel_size=1)
        self.wk = nn.Conv1d(out_feat, out_feat, kernel_size=1)
        self.wv = nn.Conv1d(out_feat, out_feat, kernel_size=1)
        
    def forward(self, xyz, feat):
        _, group_idx = pointnet2_utils.knn(self.nsample, xyz.permute(0 ,2, 1).contiguous(), xyz.permute(0 ,2, 1).contiguous()) #[B, N, K]
        
        new_feat = self.pre_fc(feat)
        pos_code = self.pos_fc(pointnet2_utils.grouping_operation(xyz, group_idx) - xyz.unsqueeze(3)) #[B, C, N, K]
        
        q = self.wq(new_feat)
        k = pointnet2_utils.grouping_operation(self.wk(new_feat), group_idx)
        v = pointnet2_utils.grouping_operation(self.wv(new_feat), group_idx)
        
        att = self.att_fc(q.unsqueeze(3) - k + pos_code)
        att = torch.softmax(att, dim=3)
        
        out = torch.sum(att * v, dim=3) #[B, C, N]
        out = self.post_fc(out + new_feat)
        
        return out
        
            
class SetInterp(nn.Module):
    def __init__(self, nsample, weight_feat, interp_feat):
        super(SetInterp, self).__init__()
        self.nsample = nsample
        self.weight_net = nn.Sequential(nn.Conv2d(2 * weight_feat + 3, weight_feat, kernel_size =1),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv2d(weight_feat, interp_feat, kernel_size=1))
        
    def forward(self, xyz1, xyz2, feat1, feat2, value1):
        '''
        xyz2 is the queries and xyz1 is the targets
        '''
        B, _, N1 = xyz1.size()
        _, _, N2 = xyz2.size()
        ex_xyz1 = xyz1
        ex_xyz2 = xyz2

        ex_xyz1_t = ex_xyz1.permute(0, 2, 1).contiguous()
        ex_xyz2_t = ex_xyz2.permute(0, 2, 1).contiguous() 
        
        _, idx = pointnet2_utils.knn(self.nsample, ex_xyz2_t, ex_xyz1_t)
        group_value = pointnet2_utils.grouping_operation(value1, idx) #[B, Cv, N2, S]
        group_xyz = pointnet2_utils.grouping_operation(xyz1, idx) - xyz2.unsqueeze(3)
        group_feat = pointnet2_utils.grouping_operation(feat1, idx)
        group_feat = torch.cat([group_feat, feat2.unsqueeze(3).repeat(1, 1, 1, self.nsample)], dim=1)
        
        weights = self.weight_net(torch.cat([group_feat, group_xyz], dim=1)) #[B, Cv, N2, S]
        weights = torch.softmax(weights, dim=3)

        interp_value = torch.sum(group_value * weights, dim=3) #[B, Cv, N2]
        
        return interp_value
    
class SetGRU(nn.Module):
    def __init__(self, radius, npoint, nsample, in_feat, hid_feat):
        super(SetGRU, self).__init__()
        self.radius = radius
        self.nsamle = nsample
        self.npoint = npoint
        
        self.convz = SetConv(radius, npoint, nsample, hid_feat + in_feat, [hid_feat], last_act=False)
        self.convr = SetConv(radius, npoint, nsample, hid_feat + in_feat, [hid_feat], last_act=False)
        self.convq = SetConv(radius, npoint, nsample, hid_feat + in_feat, [hid_feat], last_act=False)
            
    def forward(self, xyz, feat, hid_state):
        hid_feat = torch.cat([feat, hid_state], dim=1)
        
        z = torch.sigmoid(self.convz(xyz, hid_feat)[0])
        r = torch.sigmoid(self.convr(xyz, hid_feat)[0])
        q = torch.tanh(self.convq(xyz,  torch.cat([feat, r * hid_state], dim=1))[0])
        
        next_hid_state = (1 - z) * hid_state + z * q
        
        return next_hid_state
    
class SetUpdate_rec2_flow(nn.Module):
    def __init__(self, npoint, nsample, in_feat, hid_feat):
        super(SetUpdate_rec2_flow, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.in_feat = in_feat
        self.hid_feat = hid_feat
        
        self.flow_layer = SetConv(0, self.npoint, self.nsample, 3, [64, 64])
        self.align_layer = SetInterp(self.nsample, in_feat, hid_feat)
        self.iter_layer = SetGRU(0, self.npoint, self.nsample, hid_feat+64, hid_feat)
        
    def forward(self, xyz0, xyz1, state, corr0, feat0, feat1, flow0):
        B, _, N = xyz1.size()
        
        if(xyz0 is None or state is None):
            next_state = torch.zeros([B, self.hid_feat, N]).to(xyz1.device)
        else:
            if(flow0 is None):
                flow0 = torch.zeros([B, 3, N]).to(xyz1.device)       
            flow_feat0 = self.flow_layer(xyz0, flow0.contiguous())[0]
            next_state = self.iter_layer(xyz0, torch.cat([corr0, flow_feat0], dim=1), state)
            next_state = self.align_layer(xyz0, xyz1, feat0, feat1, next_state)
                
        return next_state

class DiffusionLayer_rec2_transformer(nn.Module):
    def __init__(self, radius, npoint, nsample, nupsample, naggre, condi_dim, featd_dim, feat_dim):
        super(DiffusionLayer_rec2_transformer, self).__init__()
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        self.nupsample = nupsample
        self.naggre = naggre
        self.clamp = [-200, 200]
        
        self.condi_layer = SetConv(radius, npoint, nsample, condi_dim + 3 + 3, [feat_dim, feat_dim])
        if(featd_dim > 0):
            self.upsample_layer = SetUpConv(0, nupsample, featd_dim, feat_dim, [featd_dim], [feat_dim])
        self.aggre_alyer = PointTransformer(naggre, feat_dim, feat_dim)
        self.estim_layer = nn.Sequential(Conv1d(feat_dim + 64, 64, kernel_size=1),
                                         nn.Conv1d(64, 64, kernel_size=1),
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv1d(64, 3, kernel_size=1))
        
        ### diffusion ###
        timesteps = 1000
        sampling_timesteps = 1
        self.timesteps = timesteps
        # define beta schedule
        betas = diffusion_utils.cosine_beta_schedule(timesteps=timesteps).float()
        # sampling related parameters
        self.sampling_timesteps = diffusion_utils.default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = 1.0
        self.iter = 1

        time_dim = 64
        fourier_dim = 16
        sinu_pos_emb = diffusion_utils.SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = diffusion_utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = diffusion_utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (diffusion_utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                diffusion_utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def forward(self, xyz1, xyz1d, feat, featd, flowd_up, flow_gt):
        ### flow_gt: [B, 3, N]
        ### mask: [B, 1, N]
        B, _, N = feat.size()
        
        if(flowd_up is None):
            flowd_up = torch.zeros([B, 3, N]).to(feat.device)

        if(self.training):
            gt_delta_flow = flow_gt - flowd_up
            gt_delta_flow = torch.where(torch.isinf(gt_delta_flow), torch.zeros_like(gt_delta_flow), gt_delta_flow)
            gt_delta_flow = gt_delta_flow.detach()
            
            t = torch.randint(0, self.timesteps, (B,), device=feat.device).long()
            time_feat = self.time_mlp(t) #[B, Ct]
            time_feat = time_feat.unsqueeze(2).repeat(1, 1, N) #[B, Ct, N]
            
            noise = (self.scale * torch.randn_like(gt_delta_flow)).float()
            delta_flow = self.q_sample(x_start=gt_delta_flow, t=t, noise=noise)
            
            loss = 0
            for i in range(self.iter):
                delta_flow = delta_flow.detach()
                flow_new = flowd_up + delta_flow
                
                geo_feat, _, _ = self.condi_layer(xyz1 + flow_new, torch.cat([feat, flowd_up, delta_flow], dim=1))
                if(featd is not None):
                    geo_feat = self.upsample_layer(xyz1d, xyz1, featd, geo_feat)
                geo_feat = self.aggre_alyer(xyz1 + flow_new, geo_feat)
                delta_flow = self.estim_layer(torch.cat([geo_feat, time_feat], dim=1))
                delta_flow = delta_flow.clamp(self.clamp[0], self.clamp[1])
                
                cur_loss = torch.sum((delta_flow - gt_delta_flow)**2, dim=1, keepdim=True) #[B, 1, N]
                loss += torch.sum(cur_loss.squeeze(1), dim=1) #[B]
                
            flow_new = flowd_up + delta_flow 
            
            return geo_feat, flow_new, loss
    
        else:
            device = xyz1.device
            times = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1) 
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            
            img = (self.scale * torch.randn_like(flowd_up)).float()
            for time, time_next in time_pairs:
                t = torch.full((B,), time, device=device, dtype=torch.long)
                time_feat = self.time_mlp(t) #[B, Ct]
                time_feat = time_feat.unsqueeze(2).repeat(1, 1, N) #[B, Ct, N]
                
                delta_flow = img              
                
                for i in range(self.iter):
                    delta_flow = delta_flow.detach()
                    flow_new = flowd_up + delta_flow
                    
                    geo_feat, _, _ = self.condi_layer(xyz1 + flow_new, torch.cat([feat, flowd_up, delta_flow], dim=1))
                    if(featd is not None):
                        geo_feat = self.upsample_layer(xyz1d, xyz1, featd, geo_feat)
                    geo_feat = self.aggre_alyer(xyz1 + flow_new, geo_feat)
                    delta_flow = self.estim_layer(torch.cat([geo_feat, time_feat], dim=1))
                    delta_flow = delta_flow.clamp(self.clamp[0], self.clamp[1])                
                    
                flow_new = flowd_up + delta_flow
                pred_noise = self.predict_noise_from_start(img, t, delta_flow)
                
                if(time_next < 0):
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                
                sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale * torch.randn_like(flowd_up)).float()
                
                img = delta_flow * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise
                      
            return geo_feat, flow_new, torch.zeros([B]).to(xyz1.device)
 
class DiffusionLayer_rec2_transformer_mask(nn.Module):
    def __init__(self, radius, npoint, nsample, nupsample, naggre, condi_dim, featd_dim, feat_dim):
        super(DiffusionLayer_rec2_transformer_mask, self).__init__()
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        self.nupsample = nupsample
        self.naggre = naggre
        self.clamp = [-200, 200]
        
        self.condi_layer = SetConv(radius, npoint, nsample, condi_dim + 3 + 3, [feat_dim, feat_dim])
        if(featd_dim > 0):
            self.upsample_layer = SetUpConv(0, nupsample, featd_dim, feat_dim, [featd_dim], [feat_dim])
        self.aggre_alyer = PointTransformer(naggre, feat_dim, feat_dim)
        self.estim_layer = nn.Sequential(Conv1d(feat_dim + 64, 64, kernel_size=1),
                                         nn.Conv1d(64, 64, kernel_size=1),
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv1d(64, 3, kernel_size=1))
        
        ### diffusion ###
        timesteps = 1000
        sampling_timesteps = 1
        self.timesteps = timesteps
        # define beta schedule
        betas = diffusion_utils.cosine_beta_schedule(timesteps=timesteps).float()
        # sampling related parameters
        self.sampling_timesteps = diffusion_utils.default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = 1.0
        self.iter = 1

        time_dim = 64
        fourier_dim = 16
        sinu_pos_emb = diffusion_utils.SinusoidalPosEmb(fourier_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = diffusion_utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = diffusion_utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (diffusion_utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                diffusion_utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def forward(self, xyz1, xyz1d, feat, featd, flowd_up, flow_gt, mask):
        ### flow_gt: [B, 3, N]
        ### mask: [B, 1, N]
        B, _, N = feat.size()
        
        if(flowd_up is None):
            flowd_up = torch.zeros([B, 3, N]).to(feat.device)

        if(self.training):
            gt_delta_flow = flow_gt - flowd_up
            gt_delta_flow = torch.where(torch.isinf(gt_delta_flow), torch.zeros_like(gt_delta_flow), gt_delta_flow)
            gt_delta_flow = gt_delta_flow.detach()
            
            t = torch.randint(0, self.timesteps, (B,), device=feat.device).long()
            time_feat = self.time_mlp(t) #[B, Ct]
            time_feat = time_feat.unsqueeze(2).repeat(1, 1, N) #[B, Ct, N]
            
            noise = (self.scale * torch.randn_like(gt_delta_flow)).float()
            delta_flow = self.q_sample(x_start=gt_delta_flow, t=t, noise=noise)
            
            loss = 0
            for i in range(self.iter):
                delta_flow = delta_flow.detach()
                flow_new = flowd_up + delta_flow
                
                geo_feat, _, _ = self.condi_layer(xyz1 + flow_new, torch.cat([feat, flowd_up, delta_flow], dim=1))
                if(featd is not None):
                    geo_feat = self.upsample_layer(xyz1d, xyz1, featd, geo_feat)
                geo_feat = self.aggre_alyer(xyz1 + flow_new, geo_feat)
                delta_flow = self.estim_layer(torch.cat([geo_feat, time_feat], dim=1))
                delta_flow = delta_flow.clamp(self.clamp[0], self.clamp[1])
                
                cur_loss = torch.sum((delta_flow - gt_delta_flow)**2, dim=1, keepdim=True) #[B, 1, N]
                cur_loss = cur_loss * mask
                loss += torch.sum(cur_loss.squeeze(1), dim=1) #[B]
                
            flow_new = flowd_up + delta_flow 
            
            return geo_feat, flow_new, loss
    
        else:
            device = xyz1.device
            times = torch.linspace(-1, self.timesteps - 1, steps=self.sampling_timesteps + 1) 
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            
            img = (self.scale * torch.randn_like(flowd_up)).float()
            for time, time_next in time_pairs:
                t = torch.full((B,), time, device=device, dtype=torch.long)
                time_feat = self.time_mlp(t) #[B, Ct]
                time_feat = time_feat.unsqueeze(2).repeat(1, 1, N) #[B, Ct, N]
                
                delta_flow = img              
                
                for i in range(self.iter):
                    delta_flow = delta_flow.detach()
                    flow_new = flowd_up + delta_flow
                    
                    geo_feat, _, _ = self.condi_layer(xyz1 + flow_new, torch.cat([feat, flowd_up, delta_flow], dim=1))
                    if(featd is not None):
                        geo_feat = self.upsample_layer(xyz1d, xyz1, featd, geo_feat)
                    geo_feat = self.aggre_alyer(xyz1 + flow_new, geo_feat)
                    delta_flow = self.estim_layer(torch.cat([geo_feat, time_feat], dim=1))
                    delta_flow = delta_flow.clamp(self.clamp[0], self.clamp[1])                
                    
                flow_new = flowd_up + delta_flow
                pred_noise = self.predict_noise_from_start(img, t, delta_flow)
                
                if(time_next < 0):
                    continue
                
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                
                sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale * torch.randn_like(flowd_up)).float()
                
                img = delta_flow * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise
                      
            return geo_feat, flow_new, torch.zeros([B]).to(xyz1.device)
    
class FlowEstimationLayer(nn.Module):
    def __init__(self, radius, npoint, nsample, in_feat_dim, hid_feat_dim):
        super(FlowEstimationLayer, self).__init__()
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        
        #self.aggre_layer = SetConv(radius, npoint, nsample, in_feat_dim, [hid_feat_dim, hid_feat_dim])
        self.aggre_layer1 = SetConv(radius, npoint, nsample, in_feat_dim, [hid_feat_dim])
        self.aggre_layer2 = SetConv(radius, npoint, nsample, hid_feat_dim, [hid_feat_dim])
        self.estim_layer = nn.Sequential(nn.Conv1d(hid_feat_dim, 64, kernel_size=1),
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv1d(64, 3, kernel_size=1))
        
    def forward(self, xyz, feat):
        
        new_feat1 = self.aggre_layer1(xyz, feat)[0]
        new_feat2 = self.aggre_layer2(xyz, new_feat1)[0]
        flow = self.estim_layer(new_feat2 + new_feat1)
        
        return new_feat2 + new_feat1, flow

class FlowEstimationLayer_flow(nn.Module):
    def __init__(self, radius, npoint, nsample, in_feat_dim, hid_feat_dim):
        super(FlowEstimationLayer_flow, self).__init__()
        self.radius = radius
        self.npoint = npoint
        self.nsample = nsample
        
        self.aggre_layer1 = SetConv(radius, npoint, nsample, in_feat_dim+3, [hid_feat_dim])
        self.aggre_layer2 = SetConv(radius, npoint, nsample, hid_feat_dim, [hid_feat_dim])
        self.estim_layer = nn.Sequential(nn.Conv1d(hid_feat_dim, 64, kernel_size=1),
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv1d(64, 3, kernel_size=1))
        
    def forward(self, xyz, feat, pre_flow=None):
        if(pre_flow is None):
            pre_flow = torch.zeros_like(xyz).to(xyz.device)
        feat_flow = torch.cat([feat, pre_flow], dim=1)
        new_feat1 = self.aggre_layer1(xyz, feat_flow)[0]
        new_feat2 = self.aggre_layer2(xyz, new_feat1)[0]
        flow = self.estim_layer(new_feat2 + new_feat1)

        flow = pre_flow + flow
        
        return new_feat2 + new_feat1, flow

