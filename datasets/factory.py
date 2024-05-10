import os
import pandas as pd 
from glob import glob 
from torchvision import datasets
from .augmentation import train_augmentation, test_augmentation, gt_augmentation  
from .mvtecad import MVTecAD
from .mvtecad_loco import MVTecLoco
from .visa import VISA
from .cifar10 import CIFAR10 
from .stats import datasets
from .btad import BTAD 
from .mpdd import MPDD 

def create_dataset(dataset_name:str, datadir:str, class_name:str,
                   img_size:int , mean:list , std:list, aug_info:bool = None,
                    **params):
    trainset, testset  = eval(f"load_{dataset_name}")(
                                                    dataset_name    = dataset_name,
                                                    datadir         = datadir,
                                                    class_name      = class_name,
                                                    img_size        = img_size,
                                                    mean            = mean,
                                                    std             = std,
                                                    aug_info        = aug_info, 
                                                    **params
                                                )
        
    return trainset, testset 

def load_MPDD(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_mpdd_df(
            datadir       = datadir,
            dataset_name  = dataset_name,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = MPDD(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )

    testset = MPDD(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )
    
    return trainset, testset 

def load_BTAD(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_btad_df(
            datadir       = datadir,
            dataset_name  = dataset_name,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = BTAD(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )

    testset = BTAD(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )
    
    return trainset, testset 

def load_VISA(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_visa_df(
            dataset_name  = dataset_name,
            datadir       = datadir,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = VISA(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )

    testset = VISA(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )
    
    return trainset, testset 

def load_MVTecLoco(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_df(
            dataset_name  = dataset_name,
            datadir       = datadir,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = MVTecLoco(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = ['Resize']),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )

    testset = MVTecLoco(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = ['Resize']),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )
    
    return trainset, testset   

def load_MVTecAD(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_df(
            dataset_name  = dataset_name,
            datadir       = datadir,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = MVTecAD(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )

    testset = MVTecAD(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )
    
    return trainset, testset 
        
        
def load_CIFAR10(dataset_name: str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, anomaly_ratio: float = 0.0, baseline: bool = True):    
    def class_to_idx(class_name):
        class_to_idx = {'airplane': 0,
                        'automobile': 1,
                        'bird': 2,
                        'cat': 3,
                        'deer': 4,
                        'dog': 5,
                        'frog': 6,
                        'horse': 7,
                        'ship': 8,
                        'truck': 9
                        }
        
        return class_to_idx[class_name]
    
    class_name = class_to_idx(class_name) # class_name to idx
    
    trainset = CIFAR10(
        root             = datadir,
        class_name       = class_name,
        anomaly_ratio    = anomaly_ratio,
        baseline         = baseline,
        train            = True,
        transform        = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
        target_transform = None,
        download         = True,
    )
    
    testset = CIFAR10(
        root             = datadir,
        class_name       = class_name,
        anomaly_ratio    = anomaly_ratio,
        baseline         = baseline,
        train            = False,
        transform        = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
        target_transform = None,
        download         = True,
    )
    return trainset, testset 


def get_visa_df(dataset_name: str, datadir: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    visa_dirs = pd.read_csv(os.path.join(datadir,dataset_name,'split_csv/1cls.csv'))
    visa_dirs = visa_dirs[visa_dirs['object']==class_name].reset_index(drop=True)                             
    visa_dirs = visa_dirs.rename(columns={'split':'train/test','label':'anomaly',
                                          'image':0})
    visa_dirs['anomaly'] = visa_dirs['anomaly'].apply(lambda x : 0 if x == 'normal' else 1)
    visa_dirs[0] = visa_dirs[0].apply(lambda x : os.path.join(datadir,'VISA',x))
    visa_dirs['mask'] = visa_dirs['mask'].apply(lambda x : os.path.join(datadir,'VISA',str(x)))
    
    if baseline:
        df = visa_dirs 
    else:
        df = train_test_split(
            df = visa_dirs,
            max_ratio =  0.05,
            anomaly_ratio = anomaly_ratio
        )
    return df 
        


def get_df(dataset_name: str, datadir: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    '''
    args:
        datadir : root of data 
        class_name : the name of category 
        baseline : dataset for reproduce performance of baseline if True or dataset for experiment of fully unsupervised 
    Example:
        df = get_df(
                datadir       = './Data' , 
                class_name    = 'toothbrush'
            ) 
    '''
    # get img_dirs dataframe 
    img_dirs = get_img_dirs(dataset_name=dataset_name, datadir=datadir, class_name=class_name)
    img_dirs['train/test'] = img_dirs[0].apply(lambda x : x.split('/')[-3]) # allocate initial train/test label 
    if baseline:
        df = img_dirs
    else:
        # train test split
        df = train_test_split(                
            df      = img_dirs,
            anomaly_ratio = anomaly_ratio
            )
    return df 
    

def get_img_dirs(dataset_name: str, datadir:str, class_name:str) -> pd.DataFrame:
    '''
        디렉토리 내 이미지 디렉토리들을 가져오는 메소드 
    '''
    class_name = '*' if class_name =='all' else class_name 
    
    img_dirs = pd.Series(sorted(glob(os.path.join(datadir,dataset_name, class_name,'*/*/*.png'))))
    img_dirs = pd.DataFrame(img_dirs[img_dirs.apply(lambda x : x.split('/')[-3]) != 'ground_truth']).reset_index(drop=True)
    img_dirs['anomaly'] = img_dirs[0].apply(lambda x : 1 if x.split('/')[-2] != 'good' else 0)
    img_dirs['train/test'] = ''
    return img_dirs 

def train_test_split(df, max_ratio = 0.2, anomaly_ratio=0.2):
    
    num_train = len(df[df['train/test'] == 'train'])
    num_max_anomaly = int(num_train * max_ratio)
    num_anomaly_train = int(num_train*anomaly_ratio)

    # test 데이터 중 train으로 사용 될 후보군 sampling 
    unfix_anomaly_index = df[(df['train/test'] == 'test') & (df['anomaly'] == 1)].sample(num_max_anomaly).index
    df.loc[unfix_anomaly_index, 'train/test'] = 'unfix'

    # train으로 사용될 후보군 제외 후 test 셋 고정 
    fix_test = df[(df['train/test'] == 'test')].reset_index(drop=True)

    # train의 anomaly 수 만큼 sampling 후 trainset에서 제외 
    notuse_train_normal = df[df['train/test']=='train'].sample(num_anomaly_train).index
    df.loc[notuse_train_normal,'train/test'] = 'notuse'

    # train 의 anomaly 수 만큼 후보군에서 sampling 및 train에 추가 
    train_anomaly_index = df[df['train/test']=='unfix'].sample(num_anomaly_train).index
    df.loc[train_anomaly_index,'train/test'] = 'train'

    # train 셋 고정 
    fix_train = df[df['train/test']=='train'].reset_index(drop=True)

    final_df = pd.concat([fix_train, fix_test]).reset_index(drop=True)
    
    return final_df 


def get_btad_df(datadir: str, dataset_name: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    '''
    args:
        datadir : root of data 
        class_name : the name of category 
        baseline : dataset for reproduce performance of baseline if True or dataset for experiment of fully unsupervised 
    Example:
        df = get_btad_df(
                ddataset_name = 'BTAD'
                datadir       = './Data' , 
                class_name    = 'toothbrush'
            ) 
    '''
    # get img_dirs dataframe 
    
    img_dirs = pd.Series(sorted(glob(os.path.join(datadir,dataset_name,str(class_name),'*','*','*'))))
    img_dirs = pd.DataFrame(img_dirs[img_dirs.apply(lambda x : x.split('/')[-3]) != 'ground_truth']).reset_index(drop=True)
    img_dirs['train/test'] = img_dirs[0].apply(lambda x : x.split('/')[-3]) # allocate initial train/test label 
    img_dirs['anomaly'] = img_dirs[0].apply(lambda x : 1 if x.split('/')[-2] != 'ok' else 0)
    if baseline:
        df = img_dirs
    return df 

def get_mpdd_df(datadir: str, dataset_name: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    '''
    args:
        datadir : root of data 
        class_name : the name of category 
        baseline : dataset for reproduce performance of baseline if True or dataset for experiment of fully unsupervised 
    Example:
        df = get_btad_df(
                ddataset_name = 'BTAD'
                datadir       = './Data' , 
                class_name    = 'toothbrush'
            ) 
    '''
    # get img_dirs dataframe 
    
    img_dirs = pd.Series(sorted(glob(os.path.join(datadir,dataset_name,str(class_name),'*','*','*'))))
    img_dirs = pd.DataFrame(img_dirs[img_dirs.apply(lambda x : x.split('/')[-3]) != 'ground_truth']).reset_index(drop=True)
    img_dirs['train/test'] = img_dirs[0].apply(lambda x : x.split('/')[-3]) # allocate initial train/test label 
    img_dirs['anomaly'] = img_dirs[0].apply(lambda x : 1 if x.split('/')[-2] != 'good' else 0)
    if baseline:
        df = img_dirs
    return df 