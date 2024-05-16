from omegaconf import OmegaConf
import argparse
from datasets import stats
from easydict import EasyDict
import os 


def convert_type(value):
    # None
    if value == 'None':
        return None
    
    # list or tuple
    elif len(value.split(',')) > 1:
        return value.split(',')
    
    # bool
    check, value = str_to_bool(value)
    if check:
        return value
    
    # float
    check, value = str_to_float(value)
    if check:
        return value
    
    # int
    check, value = str_to_int(value)
    if check:
        return value
    
    return value

def str_to_bool(value):
    try:
        check = isinstance(eval(value), bool)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_float(value):
    try:
        check = isinstance(eval(value), float)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_int(value):
    try:
        check = isinstance(eval(value), int)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def cli_parser():
    args = OmegaConf.from_cli()
    default_cfg = OmegaConf.load(args.default_setting)
    model_cfg = OmegaConf.load(args.model_setting)
    cfg = OmegaConf.merge(default_cfg,model_cfg)
    cfg = OmegaConf.merge(cfg,args)

    return cfg 

def jupyter_parser(default_setting:str = None, model_setting:str = None):
    default = OmegaConf.load(default_setting)
    model = OmegaConf.load(model_setting)
    return OmegaConf.merge(default, model)

def parser(jupyter:bool = False, default_setting:str = None, model_setting:str = None):
    
    if jupyter:
        cfg = jupyter_parser(default_setting, model_setting)
    else:
        cfg = cli_parser()
    
    # Update experiment name
    if cfg.MODEL.method in ['PatchCore','SoftPatch']:
        # cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-coreset_ratio_{cfg.MODEL.params.coreset_sampling_ratio}-anomaly_ratio_{cfg.DATASET.params.anomaly_ratio}" 
        cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-{cfg.MODEL.params.weight_method}-sampling_ratio_{cfg.MODEL.params.sampling_ratio}-anomaly_ratio_{cfg.DATASET.params.anomaly_ratio}" 
    else:
        cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-anomaly_ratio_{cfg.DATASET.params.anomaly_ratio}" 
               
    # load dataset statistics
    if cfg.DATASET.dataset_name in ['MVTecAD','MVTecLoco','VISA','BTAD','MPDD']:
        cfg.DATASET.update(stats.datasets['ImageNet'])
    else:    
        cfg.DATASET.update(stats.datasets[cfg.DATASET.dataset_name])    
    
    # update config for each method 
    
    if cfg.MODEL.method in ['PatchCore','ReconPatch','ProxyCoreBase','SoftPatch','CoreInit']:
        cfg = patchcore_arguments(cfg)
    else:
        pass
    
    # Print Experiment name 
    print(f"\n Experiment Name : {cfg.DEFAULT.exp_name}\n")
    
    return cfg  

def patchcore_arguments(cfg):
    # device for Patchcore 
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
        cfg.MODEL.params.device = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', None)}"
    else:
        cfg.MODEL.params.device = 'cuda:0'    

    # threshold for each sampling method 
    if cfg.MODEL.params.weight_method == 'identity':
        cfg.MODEL.params.threshold = 1 
        
    return cfg 
