import logging
import time
import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict

from utils.metrics import MetricCalculator
from utils.log import AverageMeter,metric_logging

_logger = logging.getLogger('train')


def train(model, dataloader, optimizer, accelerator, log_interval: int) -> dict:
    print('Train Start')
   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    image_bank = [] 
    for idx, (images, _, _) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        # predict
        image_bank.append(images.detach().cpu())    
        
        # batch time
        batch_time_m.update(time.time() - end)

        if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
            _logger.info('TRAIN [{:>4d}/{}]'
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1)//accelerator.gradient_accumulation_steps, 
                        len(dataloader)//accelerator.gradient_accumulation_steps, 
                        batch_time = batch_time_m,
                        rate       = images[0].size(0) / batch_time_m.val,
                        rate_avg   = images[0].size(0) / batch_time_m.avg,
                        data_time  = data_time_m
                        ))                
        end = time.time()
        
    model.fit(image_bank)

def test(model, dataloader) -> dict:    
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])
        
    for idx, (images, labels, gts) in enumerate(dataloader):
        
        # predict
        score, score_map = model.predict(images)
        score = np.array(score)
        score_map = np.concatenate([np.expand_dims(sm,0) for sm in score_map]) #(B,W,H)
        score_map = np.expand_dims(score_map,1) # (B,1,W,H)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
        
    # Calculate results of evaluation     
    if dataloader.dataset.name != 'CIFAR10':
        p_results = pix_level.compute()
    i_results = img_level.compute()
    
    # Calculate results of evaluation per each images        
    if dataloader.dataset.__class__.__name__ == 'MVTecLoco':
        p_results['loco_auroc'] = loco_auroc(pix_level,dataloader)
        i_results['loco_auroc'] = loco_auroc(img_level,dataloader)                
            
    # logging metrics
    if dataloader.dataset.name != 'CIFAR10':
        _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
    else:
        _logger.info('Image AUROC: %.3f%%' % (i_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    if dataloader.dataset.name != 'CIFAR10':
        test_result.update([('pix_level', p_results)])
    
    return test_result 


def fit(
    model, trainloader, testloader, optimizer, scheduler, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    

        
    train(
        model        = model, 
        dataloader   = trainloader, 
        optimizer    = optimizer, 
        accelerator  = accelerator, 
        log_interval = log_interval
    )        
    
    test_metrics = test(
        model        = model, 
        dataloader   = testloader
    )
    
    epoch_time_m.update(time.time() - end)
    end = time.time()
    
    
    # logging 
    metric_logging(
        savedir = savedir, use_wandb = use_wandb, epoch = 1, step = 1,
        optimizer = optimizer, epoch_time_m = epoch_time_m,
        train_metrics = None, test_metrics = test_metrics)
        
                
    # checkpoint - save best results and model weights        
    if best_score < test_metrics['img_level']['auroc']:
        best_score = test_metrics['img_level']['auroc']
        print(f" New best score : {best_score} | best epoch : {1}\n")
        

