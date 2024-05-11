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
    
    end = time.time()
    
    model.eval()
    model.fit(dataloader)
    _logger.info(f"Train Time: {time.time() - end}")

def test(model, dataloader) -> dict:    
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])
        
    for idx, (images, labels, gts) in enumerate(dataloader):
        
        # predict
        with torch.no_grad():
            score, score_map = model.predict(images)
                
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
    ,cfg=None):

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
        

