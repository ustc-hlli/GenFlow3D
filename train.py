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
    model = models.GenFlow_rec_seq(args.npoint)
    
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
        print('Training from scratch')
        logger.info('Training from scratch')
        init_epoch = 0
    else:
        if(args.multi_gpu):
            model.module.load_state_dict(torch.load(args.pretrain))
        else:
            model.load_state_dict(torch.load(args.pretrain))
        print('Load checkpoint: %s, training from epoch %d' % (args.pretrain, args.init_epoch))
        logger.info('Load checkpoint: %s, training from epoch %d' % (args.pretrain, args.init_epoch))
        init_epoch = args.init_epoch
    
    if(args.optimizer == 'SGD'):
        optimizer = torch.optim.SGD([{'params':model.parameters(), 'initial_lr':args.learning_rate}], lr=args.learning_rate, momentum=0.9)
    elif(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam([{'params':model.parameters(), 'initial_lr':args.learning_rate}], lr=args.learning_rate, betas=(0.9, 0.999),
                                      eps=1e-8, weight_decay=args.weight_decay)
    elif(args.optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW([{'params':model.parameters(), 'initial_lr':args.learning_rate}], lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    else:
        logger.info('Not implemented optimizer...')
        raise ValueError('Not implemented optimizer')
        
    if(args.dataset.lower() == 'nuscenes'):
        train_set = datasets.NuScenes_seq(args.data_root,
                                     args.npoint,
                                     args.length_pre,
                                     args.length_fut,
                                     True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=16, pin_memory=True, drop_last=False,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        
        eval_set = datasets.NuScenes_seq(args.data_root,
                                     args.npoint,
                                     args.length_pre,
                                     args.length_fut,
                                     False)
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=16, pin_memory=True, drop_last=False,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        logger.info('Train Dataset:' + args.dataset + 'Sample Number:' + str(len(train_set)))
        logger.info('Eval Dataset:' + args.dataset + 'Sample Number:' + str(len(eval_set)))
    else:
        logger.info('Not implemented dataset...')
        raise ValueError('Not implemented dataset')
    print('Dataset loaders ready!!!')    

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7, last_epoch = init_epoch - 1)
    
    if(args.best_epe is None):
        best_epe = np.inf
    tik = torch.cuda.Event(enable_timing=True)
    tok = torch.cuda.Event(enable_timing=True)
    for e in range(init_epoch, args.epochs):
        cur_lr = max(optimizer.param_groups[0]['lr'], 1e-5)
        print('learning rate=', cur_lr)
        logger.info('learning rate='+str(cur_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
        
        total_loss = 0
        total_seen = 0
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            tik.record()
            model = model.train()
            
            pcs, feats, flows = data
            cur_bs = flows[0].size()[0]
            pcs = [pc.cuda() for pc in pcs]
            feats = [feat.cuda() for feat in feats]
            flows = [flow.cuda() for flow in flows]
            
            optimizer.zero_grad()
            pred_flows, fps_inds, tot_loss_difs = model(pcs, feats, flows)
            loss = models.multiscalerecurrentloss_seq(flows, pred_flows, fps_inds, [0.02, 0.04, 0.08, 0.16], tot_loss_difs, args.weight_dist, args.weight_dif)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach().cpu().data * cur_bs
            total_seen += cur_bs
            
            tok.record()
            torch.cuda.synchronize()
        
        scheduler.step()
        
        print('total samples:', total_seen)
        train_loss = total_loss / total_seen
        print('EPOCH %d mean training loss: %f' % (e, train_loss))
        logger.info('EPOCH %d mean training loss: %f' % (e, train_loss))
        
        eval_loss, eval_metrics = eval_model(model, eval_loader, args)
        for jp in range(args.length_pre-1):
            print('previous frame %d, eval epe: %.5f' % (jp+1, eval_metrics['epe'][jp]))
            logger.info('previous frame %d, eval epe: %.5f' % (jp+1, eval_metrics['epe'][jp]))
        for jf in range(args.length_fut):
            print('future frame %d, eval epe: %.5f' % (jf+1, eval_metrics['epe'][jf+args.length_pre-1]))
            logger.info('future frame %d, eval epe: %.5f' % (jf+1, eval_metrics['epe'][jf+args.length_pre-1]))
            
        epe_sum = np.sum(eval_metrics['epe'])
        if(epe_sum < best_epe):
            best_epe = epe_sum
            if(args.multi_gpu):
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth' % (chk_dir, args.model_name, e, epe_sum))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth' % (chk_dir, args.model_name, e, epe_sum))
            print('Save model...')
            logger.info('Save model...')
        print('Best epe: %f' % best_epe)
    
    return None

def eval_model(model, loader, args):
    model = model.eval()
    
    total_loss = 0
    total_seen = 0
    total_metrics = {'epe': np.zeros([args.length_pre + args.length_fut - 1]),
                     'accr': np.zeros([args.length_pre + args.length_fut - 1]),
                     'accs': np.zeros([args.length_pre + args.length_fut - 1]),
                     'outliers': np.zeros([args.length_pre + args.length_fut - 1])}
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader, 0), total=len(loader)):
            pcs, feats, flows = data
            cur_bs = flows[0].size()[0]
            pcs = [pc.cuda() for pc in pcs]
            feats = [feat.cuda() for feat in feats]
            flows = [flow.cuda() for flow in flows]
            
            pred_flows, fps_inds, tot_loss_difs = model(pcs, feats, flows)
            loss = models.multiscalerecurrentloss_seq(flows, pred_flows, fps_inds, [0.02, 0.04, 0.08, 0.16], tot_loss_difs, args.weight_dist, args.weight_dif)
            metrics = utils.compute_metrics(pred_flows[0], flows)
            
            total_loss += loss.cpu().data * cur_bs
            total_seen += cur_bs
            for k in total_metrics.keys():
                total_metrics[k] += metrics[k] * cur_bs
        
        mean_loss = total_loss / total_seen
        for k in total_metrics.keys():
            total_metrics[k] /= total_seen
    return mean_loss, total_metrics

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    args = cmd_args.parse_args_from_yaml(os.path.join(root, 'train_cfg.yaml'))
    
    exp_name = args.exp_name
    exp_dir = os.path.join(root, 'experiment', exp_name)
    log_dir = os.path.join(exp_dir, 'logs')
    chk_dir = os.path.join(exp_dir, 'checkpoints')
    
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    if(not os.path.exists(chk_dir)):
        os.makedirs(chk_dir)
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
        
    files_to_save = ['models.py', 'units.py', 'layers.py', 'train.py', 'train_cfg.yaml', 'datasets.py', 'utils.py']
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