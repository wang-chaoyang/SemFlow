import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Callable, Dict, Tuple, Any, Optional,List
from .transform import RandomHorizontalFlip, CropResize, ToTensor


def get_train_transforms(p: Dict[str, Any]) -> Callable:
    size = p.size
    crop_mode = p.crop
    if size<2000:
        real_size = (size,size)
    else:
        size = size//10
        real_size = (size,2*size)
        crop_mode = None
    print('size:',real_size)
    transforms = T.Compose([
        RandomHorizontalFlip() if p.flip else nn.Identity(),
        CropResize(real_size, crop_mode=crop_mode),
        ToTensor(),
    ])
    return transforms

def get_val_transforms(p: Dict) -> Callable:
    size = p.size
    if size<2000:
        real_size = (size,size)
    else:
        size = size//10
        real_size = (size,2*size)
    print('size:',real_size)
    transforms = T.Compose([
        CropResize(real_size, crop_mode=None),
        ToTensor(),
    ])
    return transforms

def get_dataset(
    split: Any,
    db_name = 'coco',
    transform: Optional[Callable] = None,
    cfg_palette = None,
):

    args_palette = (cfg_palette.k,cfg_palette.s)

    if db_name=='celeb':
        from .celeb import Celeb
        dataset = Celeb(
            data_root='dataset/celebAmask',
            split=split,
            transform=transform,
            args_palette=args_palette
        )
    
    elif db_name=='city':
        from .cityscapes import Cityscapes
        dataset = Cityscapes(
            data_root='dataset/cityscapes',
            split=split,
            transform=transform,
            size=1024,
            args_palette=args_palette
        )

    elif db_name=='cocostuff':
        from .cocostuff import COCOStuff
        dataset = COCOStuff(
            data_root='dataset/cocostuff',
            split=split,
            transform=transform,
            args_palette=args_palette
        )

    else:
        raise NotImplementedError()

    return dataset



