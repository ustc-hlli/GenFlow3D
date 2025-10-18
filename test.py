# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:56:32 2024

@author: lihl
"""

import numpy as np
import torch
import torch.nn as nn
import os
import logging
from tqdm import tqdm

import models
import datasets
import utils
import cmd_args


def training_process(args, logger, chk_dir):
    model = model = models.GenFlow_rec_seq(args.npoint)
    
    if(args.multi_gpu):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.cuda()
        model = nn.DataParallel(model)
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.cuda()
    
    if(args.pretrain is None):
        raise ValueError('No pretrianed model')
    else:
        if(args.multi_gpu):
            ori = torch.load(args.pretrain)
            model.module.load_state_dict(ori)
        else:
            model.load_state_dict(torch.load(args.pretrain))
        print('Load checkpoint: %s' % (args.pretrain))
        logger.info('Load checkpoint: %s' % (args.pretrain))
            
    if(args.dataset.lower() == 'nuscenes'):
        eval_set = datasets.NuScenes_seq(args.data_root,
                                     args.npoint,
                                     args.length_pre,
                                     args.length_fut,
                                     False, return_fut_pc=True)
        print('N=%d, M=%d' % (args.length_pre, args.length_fut))
        logger.info('N=%d, M=%d' % (args.length_pre, args.length_fut))
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False,
                                                   num_workers=16, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        print('Eval Dataset:' + args.dataset + 'Sample Number:' + str(len(eval_set)))
        logger.info('Eval Dataset:' + args.dataset + 'Sample Number:' + str(len(eval_set)))
    else:
        logger.info('Not implemented dataset...')
        raise ValueError('Not implemented dataset')
    print('Dataset loaders ready!!!') 
   
    eval_loss, eval_metrics = eval_model(model, eval_loader, args)
    print(eval_metrics)
    logger.info(str(eval_metrics))
                
    return None

def eval_model(model, loader, args):
    model = model.eval()
    
    total_loss = 0
    total_seen = 0
    total_metrics = {'epe': np.zeros([args.length_pre + args.length_fut - 1]),
                     'accr': np.zeros([args.length_pre + args.length_fut - 1]),
                     'accs': np.zeros([args.length_pre + args.length_fut - 1]),
                     'outliers': np.zeros([args.length_pre + args.length_fut - 1]),
                     'chamfer': np.zeros([args.length_fut])}
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader, 0), total=len(loader)):
            pcs, feats, flows, fut_pcs = data
            cur_bs = flows[0].size()[0]
            pcs = [pc.cuda() for pc in pcs]
            feats = [feat.cuda() for feat in feats]
            flows = [flow.cuda() for flow in flows]
            fut_pcs = [fut_pc.cuda() for fut_pc in fut_pcs]

            pred_flows, fps_inds, tot_loss_difs= model(pcs, feats, flows)
            loss = models.multiscalerecurrentloss_seq(flows, pred_flows, fps_inds, [0.02, 0.04, 0.08, 0.16], tot_loss_difs, args.weight_dist, args.weight_dif)
            metrics = utils.compute_metrics(pred_flows[0], flows)
            fut_metrics = utils.comput_fut_metrics(pred_flows[0][args.length_pre-1:], pcs[-1], fut_pcs)
            
            total_loss += loss.cpu().data * cur_bs
            total_seen += cur_bs
            for k in metrics.keys():
                total_metrics[k] += metrics[k] * cur_bs
            for k in fut_metrics.keys():
                total_metrics[k] += fut_metrics[k] * cur_bs
        
        mean_loss = total_loss / total_seen
        for k in total_metrics.keys():
            total_metrics[k] /= total_seen

    return mean_loss, total_metrics

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    args = cmd_args.parse_args_from_yaml(os.path.join(root, 'test_cfg.yaml'))
    
    exp_name = args.exp_name
    exp_dir = os.path.join(root, 'Evaluate_experiments', exp_name)
    log_dir = os.path.join(exp_dir, 'logs')
    chk_dir = os.path.join(exp_dir, 'checkpoints')
    
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    if(not os.path.exists(chk_dir)):
        os.makedirs(chk_dir)
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
        
    files_to_save = ['models.py', 'units.py', 'layers.py', 'test.py', 'test_cfg.yaml', 'datasets.py', 'utils.py']
    for fname in files_to_save:
        os.system('cp %s %s' % (os.path.join(root, fname), log_dir))
    
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir + '/train_%s.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print('BEGIN TRAINING...')
    logger.info('BEGIN TRAINING...')
    logger.info(args)

    training_process(args, logger, chk_dir)
    print('FINISH TRAINING...')
    logger.info('FINISH TRAINING') 