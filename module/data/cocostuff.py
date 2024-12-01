import os
import os.path as osp
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from typing import Optional, Any, Tuple
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


class COCOStuff(data.Dataset):

    COCO_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
    'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
    'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
    'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
    'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
    'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
    'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
    'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
    'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
    'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
    'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
    'stone', 'straw', 'structural-other', 'table', 'tent',
    'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
    'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
    'window-blind', 'window-other', 'wood']

    class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 
                 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 
                 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 
                 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 
                 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 
                 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 
                 180, 181]
    
    mappings = {0 : 0 ,
                1 : 1 ,
                2 : 2 ,
                3 : 3 ,
                4 : 4 ,
                5 : 5 ,
                6 : 6 ,
                7 : 7 ,
                8 : 8 ,
                9 : 9 ,
                10 : 10 ,
                12 : 11 ,
                13 : 12 ,
                14 : 13 ,
                15 : 14 ,
                16 : 15 ,
                17 : 16 ,
                18 : 17 ,
                19 : 18 ,
                20 : 19 ,
                21 : 20 ,
                22 : 21 ,
                23 : 22 ,
                24 : 23 ,
                26 : 24 ,
                27 : 25 ,
                30 : 26 ,
                31 : 27 ,
                32 : 28 ,
                33 : 29 ,
                34 : 30 ,
                35 : 31 ,
                36 : 32 ,
                37 : 33 ,
                38 : 34 ,
                39 : 35 ,
                40 : 36 ,
                41 : 37 ,
                42 : 38 ,
                43 : 39 ,
                45 : 40 ,
                46 : 41 ,
                47 : 42 ,
                48 : 43 ,
                49 : 44 ,
                50 : 45 ,
                51 : 46 ,
                52 : 47 ,
                53 : 48 ,
                54 : 49 ,
                55 : 50 ,
                56 : 51 ,
                57 : 52 ,
                58 : 53 ,
                59 : 54 ,
                60 : 55 ,
                61 : 56 ,
                62 : 57 ,
                63 : 58 ,
                64 : 59 ,
                66 : 60 ,
                69 : 61 ,
                71 : 62 ,
                72 : 63 ,
                73 : 64 ,
                74 : 65 ,
                75 : 66 ,
                76 : 67 ,
                77 : 68 ,
                78 : 69 ,
                79 : 70 ,
                80 : 71 ,
                81 : 72 ,
                83 : 73 ,
                84 : 74 ,
                85 : 75 ,
                86 : 76 ,
                87 : 77 ,
                88 : 78 ,
                89 : 79 ,
                91 : 80 ,
                92 : 81 ,
                93 : 82 ,
                94 : 83 ,
                95 : 84 ,
                96 : 85 ,
                97 : 86 ,
                98 : 87 ,
                99 : 88 ,
                100 : 89 ,
                101 : 90 ,
                102 : 91 ,
                103 : 92 ,
                104 : 93 ,
                105 : 94 ,
                106 : 95 ,
                107 : 96 ,
                108 : 97 ,
                109 : 98 ,
                110 : 99 ,
                111 : 100 ,
                112 : 101 ,
                113 : 102 ,
                114 : 103 ,
                115 : 104 ,
                116 : 105 ,
                117 : 106 ,
                118 : 107 ,
                119 : 108 ,
                120 : 109 ,
                121 : 110 ,
                122 : 111 ,
                123 : 112 ,
                124 : 113 ,
                125 : 114 ,
                126 : 115 ,
                127 : 116 ,
                128 : 117 ,
                129 : 118 ,
                130 : 119 ,
                131 : 120 ,
                132 : 121 ,
                133 : 122 ,
                134 : 123 ,
                135 : 124 ,
                136 : 125 ,
                137 : 126 ,
                138 : 127 ,
                139 : 128 ,
                140 : 129 ,
                141 : 130 ,
                142 : 131 ,
                143 : 132 ,
                144 : 133 ,
                145 : 134 ,
                146 : 135 ,
                147 : 136 ,
                148 : 137 ,
                149 : 138 ,
                150 : 139 ,
                151 : 140 ,
                152 : 141 ,
                153 : 142 ,
                154 : 143 ,
                155 : 144 ,
                156 : 145 ,
                157 : 146 ,
                158 : 147 ,
                159 : 148 ,
                160 : 149 ,
                161 : 150 ,
                162 : 151 ,
                163 : 152 ,
                164 : 153 ,
                165 : 154 ,
                166 : 155 ,
                167 : 156 ,
                168 : 157 ,
                169 : 158 ,
                170 : 159 ,
                171 : 160 ,
                172 : 161 ,
                173 : 162 ,
                174 : 163 ,
                175 : 164 ,
                176 : 165 ,
                177 : 166 ,
                178 : 167 ,
                179 : 168 ,
                180 : 169 ,
                181 : 170 ,
                255 : 255}


    def __init__(
        self,
        data_root: str,
        split: str = 'val',
        transform: Optional[Any] = None,
        args_palette = None,
    ):
        print('init cocostuff, semseg')
        self.meta_data = {'category_names':self.COCO_CATEGORY_NAMES}
        self.transform = transform
        self.post_norm = Normalize()
        self.palette = build_palette(args_palette[0],args_palette[1])
        
        if split == 'train':
            _datalist = osp.join(data_root,'metas','train.txt')
        elif split == 'val':
            _datalist = osp.join(data_root,'metas','val.txt')
        _img_dir = osp.join(data_root,f'{split}2017')
        _seg_dir = osp.join(data_root,f'annotations/{split}2017')
        self.images = []
        self.semsegs = []
        with open(_datalist,'r') as f:
            for line in f:
                self.images.append(osp.join(_img_dir,line.strip()+'.jpg'))
                self.semsegs.append(osp.join(_seg_dir,line.strip()+'.png'))
        print(f'split={split}, processing {len(self.images)} images')

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

    def remap_id(self,x):
        assert len(x.shape)==2
        clslist = np.unique(x)
        newx = np.zeros_like(x)
        for _c in clslist:
            _x,_y = np.where(x==_c)
            newx[_x,_y] = self.mappings[_c]
        return newx.astype(np.uint8)


    def __getitem__(self, index):
        sample = {}
        _img = Image.open(self.images[index]).convert('RGB')
        sample['image'] = _img

        # make sure np.uint8
        gt_semseg = Image.open(self.semsegs[index])
        gt_semseg = self.remap_id(np.array(gt_semseg))
        sample['gt_semseg'] = Image.fromarray(gt_semseg)
        sample['image_semseg'] = Image.fromarray(self.prepare_pm(gt_semseg).astype(np.uint8))

        sample['text'] = ""

        # mask with ones for valid pixels
        sample['mask'] = np.ones(_img.size[::-1])
        sample['mask'] = Image.fromarray(sample['mask'])

        # meta data
        sample['meta'] = {
            'im_size': (_img.size[1], _img.size[0]),    # h,w
            'image_file': self.images[index],
            "image_id": int(os.path.basename(self.images[index]).split(".")[0]),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample = self.post_norm(sample)

        return sample


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    pass