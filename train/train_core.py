import logging
import wandb
import time
import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from collections import OrderedDict
from accelerate import Accelerator
import wandb 
from omegaconf import OmegaConf


from refinement.sampler import SubsetSequentialSampler
from refinement.refinement import Refinementer

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
    


def train(model, dataloader, optimizer, accelerator: Accelerator, log_interval: int) -> dict:
    print('Train Start')
   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()

    for idx, (images, _, _) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        # predict
        output = model(images)
        loss   = model.criterion(output)

        # loss update
        if model.__class__.__name__ not in ['PatchCore']:
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
        # Coreset Update for PatchCore

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
                            len(dataloader)//accelerator.gradient_accumulation_steps, 
                            loss       = losses_m, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = images[0].size(0) / batch_time_m.val,
                            rate_avg   = images[0].size(0) / batch_time_m.avg,
                            data_time  = data_time_m
                            ))                
        end = time.time()
        
    if model.__class__.__name__ in ['PatchCore']:
        model.fit()
        
    # logging metrics
    _logger.info('TRAIN: Loss: %.3f' % (losses_m.avg))
    
    train_result = {'loss' : losses_m.avg}
    return train_result 

def test(model, dataloader) -> dict:    
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision','confusion_matrix'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision','confusion_matrix','aupro'])
        
    for idx, (images, labels, gts) in enumerate(dataloader):
        
        # predict
        if model.__class__.__name__ in ['PatchCore']:
            score, score_map = model.get_score_map(images)
        else:
            with torch.no_grad():
                outputs = model(images)   
                score_map = model.get_score_map(outputs).detach().cpu()
                score = score_map.reshape(score_map.shape[0],-1).max(-1)[0]
                
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
    model, trainloader, testloader, optimizer, scheduler, accelerator: Accelerator,
    n_round :int, epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for step,  epoch in enumerate(range(epochs)):
        _logger.info(f'Epoch: {epoch+1}/{epochs}')
        
        train_metrics = train(
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
        
        if scheduler is not None:
            scheduler.step()
        
        # logging 
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, r = n_round, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            train_metrics = train_metrics, test_metrics = test_metrics)
        
        # if epoch%49 == 0:
        #     torch.save(model.state_dict(), os.path.join(savedir, f'model_{epoch}.pt')) 
                
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
        model          = __import__('models').__dict__[method](
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
        model = refinement.init_model()
            
        
        _logger.info(f'Round : [{r}/{nb_round}]')     
        
        # optimizer
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](model.parameters(), lr=lr, **opt_params)

        # scheduler
        if scheduler_name is not None:
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[scheduler_name](optimizer, **scheduler_params)
        else:
            scheduler = None
        
        
        # # prepraring accelerator
        model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )
        
        # initialize wandb
        if use_wandb:
            wandb.init(name=f'{exp_name}_round{r}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))

        # fitting model
        fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
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

