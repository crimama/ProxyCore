import logging
import logging.handlers
import os 
import json 
import wandb 
from collections import OrderedDict

class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        with open(log_path, 'a') as f: 
            f.write('\n')
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        
        
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
                     epoch, step,epoch_time_m,
                    optimizer,test_metrics,train_metrics=None):
    
    metrics = OrderedDict()
    metrics.update([('epoch', epoch)])
    metrics.update([('lr',round(optimizer.param_groups[0]['lr'],5))])
    if train_metrics is not None:
        metrics.update([('train_' + k, round(v,4)) for k, v in train_metrics.items()])
    metrics.update([
                    # ('test_' + k, round(v,4)) for k, v in test_metrics.items()
                    ('test_metrics',test_metrics)
                    ])
    metrics.update([('epoch time',round(epoch_time_m.val,4))])
    
    with open(os.path.join(savedir, 'result.txt'),  'a') as f:
        f.write(json.dumps(metrics) + "\n")
    
    if use_wandb:
        wandb.log(metrics, step=step)