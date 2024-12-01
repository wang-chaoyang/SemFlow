import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image

INT_MODES = {
    'image': 'bicubic',
    'panseg': 'nearest',
    'class_labels': 'nearest',
    'mask': 'nearest',
    'image_panseg': 'bilinear',
    'image_class_labels': 'bilinear',
    'image_semseg': 'bilinear',
    'gt_semseg': 'nearest'
}

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['meta', 'text']:
                    continue
                else:
                    sample[elem] = F.hflip(sample[elem])

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip(p=0.5)'
    
class CropResize(object):
    def __init__(self, size, crop_mode=None):
        self.size = size
        self.crop_mode = crop_mode
        assert self.crop_mode in ['centre', 'random', None]

    def crop_and_resize(self, img, h, w, mode='bicubic', crop_size=None):
        # crop
        if self.crop_mode == 'centre':
            img_w, img_h = img.size
            min_size = min(img_h, img_w)
            if min_size == img_h:
                margin = (img_w - min_size) // 2
                new_img = img.crop((margin, 0, margin+min_size, min_size))
            else:
                margin = (img_h - min_size) // 2
                new_img = img.crop((0, margin, min_size, margin+min_size))
        elif self.crop_mode == 'random':
            new_img = img.crop(crop_size)
        else:
            new_img = img

        # accelerate
        if new_img.size==(w,h):
            return new_img
        # resize
        if mode == 'bicubic':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)
        elif mode == 'bilinear':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BILINEAR, reducing_gap=None)
        elif mode == 'nearest':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).NEAREST, reducing_gap=None)
        else:
            raise NotImplementedError
        return new_img

    def rand_decide(self,img):
        """
        decide crop size in random mode, the crop size in a sample should be the same
        """
        img_w, img_h = img.size
        min_size = min(img_h, img_w)
        if min_size == img_h:
            margin = random.randint(0,img_w-min_size)
            return (margin, 0, margin+min_size, min_size)
        else:
            margin = random.randint(0,img_h-min_size)
            return (0, margin, min_size, margin+min_size)


    def __call__(self, sample):
        if self.crop_mode == 'random':
            crop_size = self.rand_decide(sample['image'])
        else:
            crop_size = None
        for elem in sample.keys():
            if elem in ['image', 'image_panseg', 'panseg', 'mask', 'class_labels', 'image_class_labels', 'image_semseg']:
                sample[elem] = self.crop_and_resize(sample[elem], self.size[0], self.size[1], mode=INT_MODES[elem], crop_size=crop_size)
        return sample

    def __str__(self) -> str:
        return f"CropResize(size={self.size}, crop_mode={self.crop_mode})"


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem or 'text' in elem:
                continue

            elif elem in ['image', 'image_panseg', 'image_class_labels', 'image_semseg']:
                sample[elem] = self.to_tensor(sample[elem])  # Regular ToTensor operation

            elif elem in ['panseg', 'mask', 'class_labels', 'gt_semseg']:
                sample[elem] = torch.from_numpy(np.array(sample[elem])).long()  # Torch Long

            else:
                raise NotImplementedError

        return sample

    def __str__(self):
        return 'ToTensor'



