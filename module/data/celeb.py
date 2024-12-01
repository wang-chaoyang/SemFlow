import os
import os.path as osp
import json
import torch
import time
import numpy as np
import torch.utils.data as data
from PIL import Image

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
    

class Celeb(data.Dataset):
    COCO_CATEGORY_NAMES = ['background','skin', 'nose', 'eye_g', 'l_eye', 
                           'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 
                           'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 
                           'ear_r', 'neck_l', 'neck', 'cloth']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        transform = None,
        args_palette = None,
    ):
        print('init celebA, semseg')
        _img_dir = osp.join(data_root,'CelebA-512-img')
        _seg_dir = osp.join(data_root,'CelebAMask-HQ-mergemask')
        self.meta_data = {'category_names':self.COCO_CATEGORY_NAMES}
        self.transform = transform
        self.post_norm = Normalize()

        self.palette = build_palette(args_palette[0],args_palette[1])
        
        if split == 'train':
            _datalist = osp.join(data_root,'metas','train.txt')
        elif split == 'val':
            _datalist = osp.join(data_root,'metas','val.txt')
        self.images = []
        self.semsegs = []
        with open(_datalist,'r') as f:
            for line in f:
                self.images.append(osp.join(_img_dir,line.strip()+'.jpg'))
                self.semsegs.append(osp.join(_seg_dir,line.strip()+'.png'))
        print(f'processing {len(self.images)} images')

    def __len__(self):
        return len(self.images)
    
    def prepare_pm(self,x):
        assert len(x.shape)==2
        h,w = x.shape
        pm = np.ones((h,w,3))*255
        clslist = np.unique(x).tolist()
        if 255 in clslist:
            raise ValueError()
        for _c in clslist:
            _x,_y = np.where(x==_c)
            pm[_x,_y,:] = self.palette[int(_c)*3:(int(_c)+1)*3]
        return pm
    
    def __getitem__(self, index):
        sample = {}
        _img = Image.open(self.images[index]).convert('RGB')
        sample['image'] = _img
        sample['gt_semseg'] = Image.open(self.semsegs[index])
        sample['mask'] = np.ones_like(np.array(sample['gt_semseg']))
        sample['mask'] = Image.fromarray(sample['mask'])
        sample['image_semseg'] = Image.fromarray(self.prepare_pm(np.array(sample['gt_semseg'])).astype(np.uint8))
        sample['text'] = None
        
        sample['meta'] = {
            'im_size': (_img.size[1], _img.size[0]),
            'image_file': self.images[index],
            "image_id": int(os.path.basename(self.images[index]).split(".")[0])
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample = self.post_norm(sample)

        return sample