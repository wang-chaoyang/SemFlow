import os
import os.path as osp
import json
import torch
import time
import numpy as np
import torch.utils.data as data
from PIL import Image
# from .map_tbl import palette
from .builder import build_palette


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        for k in sample.keys():
            if k in ['image', 'image_panseg','image_semseg']:
                sample[k] = (sample[k]-self.mean)/self.std
        return sample
    

class Cityscapes(data.Dataset):
    COCO_CATEGORY_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                            'traffic light', 'traffic sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        transform = None,
        size=1024,
        args_palette = None,
    ):
        
        self.palette = build_palette(args_palette[0],args_palette[1])
        
        self.size = size
        self.data_root = data_root
        self.split = split
        if split=='train':
            self.training = True
        elif split=='val':
            self.training = False
        else:
            raise ValueError()
        print(f'init cityspace, semseg, size: {size}')
        if size==1024:
            _img_dir = osp.join(data_root,'leftImg8bit')
            _seg_dir = osp.join(data_root,'gtFine')
        else:
            raise ValueError()

        self.meta_data = {'category_names':self.COCO_CATEGORY_NAMES}
        self.transform = transform
        self.post_norm = Normalize()
        
        if split == 'train':
            _datalist = osp.join(data_root,'metas','train.txt')
        elif split == 'val':
            _datalist = osp.join(data_root,'metas','val.txt')
        self.images = []
        self.semsegs = []
        with open(_datalist,'r') as f:
            for line in f:
                self.images.append(osp.join(_img_dir,split,line.strip()+'_leftImg8bit.png'))
                self.semsegs.append(osp.join(_seg_dir,split,line.strip()+'_gtFine_labelTrainIds.png'))
        print(f'processing {len(self.images)} images')


    def __len__(self):
        return len(self.images)
    
    def prepare_pm(self,x):
        assert len(x.shape)==2
        h,w = x.shape
        pm = np.ones((h,w,3))*255
        clslist = np.unique(x).tolist()
        if 255 in clslist:
            clslist.remove(255)
        for _c in clslist:
            _x,_y = np.where(x==_c)
            pm[_x,_y,:] = self.palette[int(_c)*3:(int(_c)+1)*3]
        return pm
    
    def __getitem__(self, index):
        sample = {}
        _img = Image.open(self.images[index]).convert('RGB')
        sample['gt_semseg'] = Image.open(self.semsegs[index])

        sample['image'] = _img
        
        sample['image_semseg'] = Image.fromarray(self.prepare_pm(np.array(sample['gt_semseg'])).astype(np.uint8))
        sample['text'] = None
        
        sample['meta'] = {
            'im_size': (_img.size[1], _img.size[0]),
            'image_file': self.images[index],
            "image_id": os.path.basename(self.images[index]).split(".")[0]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['mask'] = torch.ones(sample['image'].shape[1:]).long()

        sample = self.post_norm(sample)

        return sample