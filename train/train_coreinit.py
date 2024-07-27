import logging
import time
import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict

from utils.metrics import MetricCalculator, loco_auroc
from utils.log import AverageMeter,metric_logging


_logger = logging.getLogger('train')

def train(model, featureloader, optimizer, accelerator, log_interval: int, cfg: dict) -> dict:
    _logger.info('Train Start')
   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()    
    end = time.time()
    
    model.train()        
    for idx, (feat, target) in enumerate(featureloader):
        data_time_m.update(time.time() - end)
        
        #! augmentation 
        if cfg.DATASET.embed_augemtation:
            feat = feat + torch.randn(feat.shape).to(feat.device)        
        
        # predict
        outputs = model(feat.to(accelerator.device)) # outputs = [z,w]
        loss   = model.criterion([outputs, target])

        # loss update
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
            
        losses_m.update(loss.item())                            
        # batch time
        batch_time_m.update(time.time() - end)

        if (idx+1) % accelerator.gradient_accumulation_steps == 0:
            if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1)//accelerator.gradient_accumulation_steps, 
                            len(featureloader)//accelerator.gradient_accumulation_steps, 
                            loss       = losses_m, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = feat[0].size(0) / batch_time_m.val,
                            rate_avg   = feat[0].size(0) / batch_time_m.avg,
                            data_time  = data_time_m
                            ))                
        end = time.time()
    
    # logging metrics
    _logger.info('TRAIN: Loss: %.3f' % (losses_m.avg))
    
    train_result = {'loss' : losses_m.avg}
    return train_result 

def test(model, featureloader, testloader, device) -> dict:        
    model.eval()
    
    # project features to new space 
    target_oriented_train_feat = [] 
    for feat, target in featureloader:
        feat = feat.to(device)
        with torch.no_grad():
            z = model.embedding_layer(feat)
            target_oriented_train_feat.append(z.detach().cpu().numpy())            
    target_oriented_train_feat = np.vstack(target_oriented_train_feat)
    
    # coreset subsampling 
    sample_features, _ = model.core.featuresampler.run(target_oriented_train_feat)
    model.core.anomaly_scorer.fit(detection_features=[sample_features])
    
    torch.cuda.empty_cache()    
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])

    for idx, (images, labels, gts) in enumerate(testloader):

        _ = model.core.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            # create features of test images 
            features, patch_shapes = model.core._embed(images.to(device), provide_patch_shapes=True)
            features = torch.Tensor(np.vstack(features)).to(device)        
            features = model.embedding_layer(features)
            
            # predict anomaly score 
            image_scores, _, _ = model.core.anomaly_scorer.predict([features.detach().cpu().numpy()])            
            
            # get patch wise anomaly score using image score    
            patch_scores = model.core.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize 
            ) # Unfold : (B)
                        
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = model.core.anomaly_segmentor.convert_to_segmentation(patch_scores) # interpolation : (B,pw,ph) -> (B,W,H)
                        
            score_map = np.concatenate([np.expand_dims(sm,0) for sm in masks])
            score_map = np.expand_dims(score_map,1)
            
            # get image wise anomaly score 
            image_scores = model.core.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = model.core.patch_maker.score(image_scores)

        # result update 
        img_level.update(image_scores, labels.type(torch.int))
        pix_level.update(score_map, gts.type(torch.int))
        
    # Calculate results of evaluation     
    p_results = pix_level.compute()
    i_results = img_level.compute()
                    
        # Calculate results of evaluation per each images        
    if testloader.dataset.__class__.__name__ == 'MVTecLoco':
        p_results['loco_auroc'] = loco_auroc(pix_level,testloader)
        i_results['loco_auroc'] = loco_auroc(img_level,testloader)                
            
    # logging metrics
    if testloader.dataset.name != 'CIFAR10':
        _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
    else:
        _logger.info('Image AUROC: %.3f%%' % (i_results['auroc']))
                    
    # Logging                     
    _logger.info('Image AUROC: %.3f, Pixel AUROC: %.3f' % (i_results['auroc'], p_results['auroc']))        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])
    
    return test_result 


def fit(
    model, trainloader, testloader, optimizer, scheduler, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None,
    cfg = None 
    ):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    ################## core init 
    featureloader = accelerator.prepare(model.get_feature_loader(trainloader))
    features = [] 
    for feat, target in featureloader:
        features.append(feat.detach().cpu().numpy())            
    features = np.vstack(features)
    
    model.core.featuresampler.percentage = cfg.MODEL.params.pslabel_sampling_ratio
    proxy, _ = model.core.featuresampler.run(features)
    model.core.featuresampler.percentage = cfg.MODEL.params.sampling_ratio
    
    #! 가장 기본 distance matrix 계산 방법 
    _logger.info('Start Claculate Distance Matrix')        
    
    if cfg.DATASET.pseudo_label == 'coreset':
        # distmat = torch.matmul(torch.Tensor(features),torch.Tensor(proxy).T)
        proxy_label = [] 
        for feat,_ in featureloader:
            proxy_label.append(torch.matmul(feat,torch.Tensor(proxy).to(feat.device).T).argmax(dim=1))
        proxy_label = torch.concat(proxy_label)
        
    elif cfg.DATASET.pseudo_label == 'normalize_coreset':
        proxy = nn.functional.normalize(torch.Tensor(proxy),dim=1)
        #distmat = torch.matmul(torch.Tensor(features),proxy.T)
        proxy_label = [] 
        for feat,_ in featureloader:
            proxy_label.append(torch.matmul(feat,proxy.T.to(feat.device)).argmax(dim=1))
        proxy_label = torch.concat(proxy_label)
    else:
        raise NotImplementedError        
    
    
    featureloader.dataset.labels = proxy_label

    model.set_criterion(proxy)

    _logger.info('Core set Done')
        
    from adamp import AdamP 
    optimizer = AdamP(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=optimizer.param_groups[0]['weight_decay'])
    
    if cfg.SCHEDULER.name is not None:
        scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
    else:
        scheduler = None    
    
    model, optimizer, scheduler = accelerator.prepare(model,optimizer, scheduler)
    ################## 

    for step,  epoch in enumerate(range(epochs)):
        _logger.info(f'Epoch: {epoch+1}/{epochs}')
        
        train_metrics = train(
            model         = model, 
            featureloader = featureloader, 
            optimizer     = optimizer, 
            accelerator   = accelerator, 
            log_interval  = log_interval,
            cfg           = cfg 
        )                
        
        epoch_time_m.update(time.time() - end)
        end = time.time()
        
        if scheduler is not None:
            if epoch > 4:
                scheduler.step()
                
        if epoch%24 == 0:
            test_metrics = test(
                model         = model, 
                featureloader = featureloader,
                testloader    = testloader,
                device        = accelerator.device
            )
            
            metric_logging(
                savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
                optimizer = optimizer, epoch_time_m = epoch_time_m,
                train_metrics = train_metrics, test_metrics = test_metrics)
            # torch.save(model.state_dict(), os.path.join(savedir, f'model_{epoch}.pt')) 
            
        # checkpoint - save best results and model weights    
        if best_score < test_metrics['img_level']['auroc']:
            best_score = test_metrics['img_level']['auroc']
            _logger.info(f" New best score : {best_score} | best epoch : {epoch}\n")
            #torch.save(model.state_dict(), os.path.join(savedir, f'model_best.pt')) 
            
    test_metrics = test(
        model         = model, 
        featureloader = featureloader,
        testloader    = testloader,
        device        = accelerator.device
    )  
    # logging 
    metric_logging(
        savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
        optimizer = optimizer, epoch_time_m = epoch_time_m,
        train_metrics = train_metrics, test_metrics = test_metrics)
    
    