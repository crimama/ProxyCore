import logging
import wandb
import time
import os
import json
import pdb 

import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from collections import OrderedDict
from accelerate import Accelerator


from omegaconf import OmegaConf

from models import CoTeachingLoss        
from refinement.sampler import SubsetSequentialSampler
from refinement.refinement import Refinementer

from utils.metrics import MetricCalculator

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
             

def metric_logging(savedir, use_wandb, 
                    r, epoch, step,epoch_time_m,
                    optimizer, train_metrics, test_metrics):
    
    metrics = OrderedDict(round=r)
    metrics.update([('epoch', epoch)])
    metrics.update([('lr',round(optimizer.param_groups[0]['lr'],5))])
    metrics.update([('train_' + k, round(v,4)) for k, v in train_metrics.items()])
    metrics.update([
        # ('test_' + k, round(v,4)) for k, v in test_metrics.items()
        ('test_metrics',test_metrics)
        ])
    metrics.update([('epoch time',round(epoch_time_m.val,4))])
    
    with open(os.path.join(savedir, 'log.txt'),  'w') as f:
        f.write(json.dumps(metrics) + "\n")
    if use_wandb:
        wandb.log(metrics, step=step)
    


def train(model1, model2, criterion1, criterion2, featureloader, optimizer, accelerator: Accelerator, epoch, log_interval: int) -> dict:
    print('Train Start')
   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m1 = AverageMeter()
    losses_m2 = AverageMeter()    
    end = time.time()

    model1.train()    
    model2.train()
    featureloader = accelerator.prepare(featureloader)
    for idx, (feat, target) in enumerate(featureloader):

        data_time_m.update(time.time() - end)
        # predict
        output1 = model1(feat.to(accelerator.device)) 
        output2 = model2(feat.to(accelerator.device)) 
        
        loss1, _  = criterion1([output1, output2], target, epoch)
        loss2, _  = criterion2([output2, output1], target, epoch)        
        # loss update
        optimizer.zero_grad()
        accelerator.backward(loss1)
        accelerator.backward(loss2)
        optimizer.step()
            
        losses_m1.update(loss1.item())
        losses_m2.update(loss2.item())
        # batch time
        batch_time_m.update(time.time() - end)

        if (idx+1) % accelerator.gradient_accumulation_steps == 0:
            if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss1: {loss_1.val:>6.4f} ({loss_1.avg:>6.4f}) Loss2: {loss_2.val:>6.4f} ({loss_2.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1)//accelerator.gradient_accumulation_steps, 
                            len(featureloader)//accelerator.gradient_accumulation_steps, 
                            loss_1       = losses_m1, 
                            loss_2       = losses_m2, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = feat[0].size(0) / batch_time_m.val,
                            rate_avg   = feat[0].size(0) / batch_time_m.avg,
                            data_time  = data_time_m
                            ))                
        end = time.time()
    
    # logging metrics
    _logger.info('TRAIN: Loss_1: %.3f Loss_2: %.3f' % (losses_m1.avg, losses_m2.avg) )
    
    train_result = {'loss_1' : losses_m1.avg,
                    'loss_2' : losses_m2.avg
                    }
    return train_result 

def test(model, featureloader, testloader, device) -> dict:    
    
    target_oriented_train_feat = [] 
    model.eval()
    for feat,target in featureloader:
        feat = feat.to(device)
        with torch.no_grad():
            z = model.embedding_layer(feat)
            target_oriented_train_feat.append(z.detach().cpu().numpy())
            
    target_oriented_train_feat = np.vstack(target_oriented_train_feat)
    
    sample_features, _ = model.featuresampler.run(target_oriented_train_feat)
    torch.cuda.empty_cache()
    model.anomaly_scorer.fit(detection_features=[sample_features])
    
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])

    for idx, (images, labels, gts) in enumerate(testloader):

        _ = model.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = model._embed(images.to(device), provide_patch_shapes=True)
            features = torch.Tensor(np.vstack(features)).to(device)        
            features = model.embedding_layer(features)
            
            image_scores, _, indices = model.anomaly_scorer.predict([features.detach().cpu().numpy()])            
            
            patch_scores = image_scores
            
            image_scores = model.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = model.patch_maker.score(image_scores)
            
            patch_scores = model.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            ) # Unfold : (B)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = model.anomaly_segmentor.convert_to_segmentation(patch_scores) # interpolation : (B,pw,ph) -> (B,W,H)
            
            score_map = np.concatenate([np.expand_dims(sm,0) for sm in masks])
            score_map = np.expand_dims(score_map,1)

        img_level.update(image_scores, labels.type(torch.int))
        pix_level.update(score_map, gts.type(torch.int))
        
    # Calculate results of evaluation     
    p_results = pix_level.compute()
    i_results = img_level.compute()
                    
    _logger.info('Image AUROC: %.3f%%' % (i_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])
    
    return test_result 


def fit(
    model1, model2, criterion1, criterion2, trainloader, testloader, optimizer, scheduler, accelerator: Accelerator,
    n_round :int, epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    # Prepare for feature learning 
    featureloader = model1.get_feature_loader(trainloader)
    torch.cuda.empty_cache()    
    
    
    for step,  epoch in enumerate(range(epochs)):
        _logger.info(f'Epoch: {epoch+1}/{epochs}')
        
        if epoch != 0:
            pass 
        
        train_metrics = train(
            model1        = model1, 
            model2        = model2, 
            criterion1    = criterion1,
            criterion2    = criterion2,
            featureloader = featureloader, 
            optimizer     = optimizer, 
            accelerator   = accelerator, 
            log_interval  = log_interval,
            epoch         = epoch
        )                
        
        epoch_time_m.update(time.time() - end)
        end = time.time()
        
        if scheduler is not None:
            scheduler.step()                
    
    test_metrics = test(
        model         = model2, 
        featureloader = featureloader,
        testloader    = testloader,
        device        = accelerator.device
    )
    # logging 
    metric_logging(
        savedir = savedir, use_wandb = use_wandb, r = n_round, epoch = epoch, step = step,
        optimizer = optimizer, epoch_time_m = epoch_time_m,
        train_metrics = train_metrics, test_metrics = test_metrics)
    
    # checkpoint - save best results and model weights        
    if best_score < test_metrics['img_level']['auroc']:
        best_score = test_metrics['img_level']['auroc']
        print(f" New best score : {best_score} | best epoch : {epoch}\n")
        # torch.save(model.state_dict(), os.path.join(savedir, f'model_best.pt')) 


def export_query_result(query_store:list, img_dirs):
    df = pd.DataFrame(query_store)
    df.columns = ['round','query']
    df = df.explode('query').reset_index(drop=True)
    df['query'] = df['query'].map(int)
    df['query'] = df['query'].apply(lambda x : img_dirs[x])
    return df 
        
def refinement_run(
    exp_name: str, 
    method: str, backbone: str, model_params: dict,
    trainset, testset,
    nb_round: int,
    batch_size: int, test_batch_size: int, num_workers: int, 
    opt_name: str, lr: float, opt_params: dict, 
    scheduler_name: str, scheduler_params: dict, 
    epochs: int, log_interval: int, use_wandb: bool, 
    savedir: str, seed: int, accelerator: Accelerator, cfg: dict = None):
    
    assert cfg != None if use_wandb else True, 'If you use wandb, configs should be exist.'        
    
    # define train dataloader
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = batch_size,
        num_workers = num_workers,
        shuffle     = True 
    )    
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = test_batch_size,
        shuffle     = False,
        num_workers = num_workers
    )
    
        
    
    refinement = Refinementer(
        model          = __import__('models').__dict__['Proxy'](
                           backbone = backbone,
                           **model_params
                           ),
        n_query        = cfg.REFINEMENT.n_query,
        dataset        = trainset,
        unrefined_idx  = np.ones(len(trainset)).astype(np.bool8),
        batch_size     = batch_size,
        test_transform = testset.transform,
        num_workers    = num_workers
    )
    
    
    # run
    query_store = [] 
    for r in range(nb_round):
        
        if r!= 0:
            # refinement 
            query_idx = refinement.query(model)
            print(query_idx)
            query_store.append([r-1,query_idx])
            
            df = export_query_result(
                query_store = query_store,
                img_dirs    = refinement.dataset.img_dirs
                )
            
            df.to_csv(os.path.join(savedir,'query_result.csv'))
            
            del optimizer, scheduler, trainloader, model        
            accelerator.free_memory()
            
            # update query and create new trainloader  
            trainloader = refinement.update(query_idx)
            
        # build new model 
        model1 = refinement.init_model()
        model2 = refinement.init_model()            
        
        _logger.info(f'Round : [{r}/{nb_round}]')
        
        criterion1 = CoTeachingLoss(sz_embedding=1024,
                                    nb_classes=784)   
        
        criterion2 = CoTeachingLoss(sz_embedding=1024,
                                    nb_classes=784)     
        
        # optimizer
        from adamp import AdamP
        optimizer = AdamP(list(model1.parameters()) + list(model2.parameters()) + list(criterion1.parameters()) + list(criterion2.parameters()),
                          lr=lr, **opt_params)
        # scheduler
        if scheduler_name is not None:
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[scheduler_name](optimizer, **scheduler_params)
        else:
            scheduler = None        
        
        # # prepraring accelerator
        model1, model2, criterion1, criterion2, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model1, model2, criterion1, criterion2, optimizer, trainloader, testloader, scheduler
        )    

        # fitting model
        fit(
            model1       = model1, 
            model2       = model2, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            criterion1   = criterion1,
            criterion2   = criterion2,
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            n_round      = r,
            epochs       = epochs, 
            use_wandb    = use_wandb,
            log_interval = log_interval,
            savedir      = savedir ,
            seed         = seed 
        )     
        
        #torch.save(model.state_dict(),os.path.join(savedir, f'model_round{r}_last.pt'))                   
        wandb.finish()                            


