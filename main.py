import numpy as np
import os
import random
import wandb
import torch
import logging
from arguments import parser

from datasets import create_dataset
from utils.log import setup_default_logging

from accelerate import Accelerator
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from adamp import AdamP
torch.autograd.set_detect_anomaly(True)

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):
    #! DIRECTORY SETTING 
    savedir = os.path.join(
                            cfg.DEFAULT.savedir,
                            cfg.MODEL.method,
                            cfg.DATASET.dataset_name,
                            str(cfg.DATASET.class_name),
                            cfg.DEFAULT.exp_name,
                            f"seed_{cfg.DEFAULT.seed}"
                            )    
    os.makedirs(savedir, exist_ok=True)    
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    setup_default_logging(log_path=os.path.join(savedir,'train.log'))
    
    #! TRAIN SETTING 
    # set accelerator
    accelerator = Accelerator(
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # set seed 
    torch_seed(cfg.DEFAULT.seed)

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # load dataset
    trainset, testset = create_dataset(
        dataset_name  = cfg.DATASET.dataset_name,
        datadir       = cfg.DATASET.datadir,
        class_name    = cfg.DATASET.class_name,
        img_size      = cfg.DATASET.img_size,
        mean          = cfg.DATASET.mean,
        std           = cfg.DATASET.std,
        aug_info      = cfg.DATASET.aug_info,
        **cfg.DATASET.get('params',{})
    )
    
    trainloader = DataLoader(
        dataset     = trainset,
        batch_size  = cfg.DATASET.batch_size,
        num_workers = cfg.DATASET.num_workers,
        shuffle     = True 
    )    
    
    # define test dataloader
    testloader = DataLoader(
        dataset     = testset,
        batch_size  = cfg.DATASET.test_batch_size,
        num_workers = cfg.DATASET.num_workers,
        shuffle     = False
    )
    
    # model 
    model  = __import__('models').__dict__[cfg.MODEL.method](
                backbone = cfg.MODEL.backbone,
                **cfg.MODEL.params
                )
    
    # optimizer 
    # if cfg.OPTIMIZER.name is not None:
    # optimizer = __import__('torch.optim',fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name]()    
    optimizer = AdamP(model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)
    
    # scheduler 
    if cfg.SCHEDULER.name is not None:
        scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
    else:
        scheduler = None        
    
    
    if cfg.TRAIN.wandb.use:
        wandb.init(name=f'{cfg.DEFAULT.exp_name}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))
    
    # prepare 
    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, testloader, scheduler
        )
                
        
    __import__(f'train.train_{cfg.MODEL.method.lower()}', fromlist=f'train_{cfg.MODEL.method.lower()}').fit(
            model        = model, 
            trainloader  = trainloader, 
            testloader   = testloader, 
            optimizer    = optimizer, 
            scheduler    = scheduler,
            accelerator  = accelerator,
            epochs       = cfg.TRAIN.epochs, 
            use_wandb    = cfg.TRAIN.wandb.use,
            log_interval = cfg.TRAIN.log_interval,
            savedir      = savedir,
            seed         = cfg.DEFAULT.seed,
            cfg          = cfg
        )     
    

if __name__=='__main__':

    # config
    cfg = parser()
    
    # run
    run(cfg)