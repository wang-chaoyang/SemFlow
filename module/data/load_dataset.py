import torch
from .utils import get_train_transforms, get_val_transforms, get_dataset
from torch.utils.data import DataLoader

def collate_fn(batch: dict):
    # TODO: make general
    semseg = image_panseg = None
    images = torch.stack([d['image'] for d in batch])
    if 'panseg' in batch[0]:
        semseg = torch.stack([d['panseg'] for d in batch])
    if 'image_panseg' in batch[0]:
        image_panseg = torch.stack([d['image_panseg'] for d in batch])
    image_semseg = torch.stack([d['image_semseg'] for d in batch])
    gt_semseg = [d['gt_semseg'] for d in batch]
    tokens = mask = inpainting_mask = text = meta = None
    if 'tokens' in batch[0]:
        tokens = torch.stack([d['tokens'] for d in batch])
    if 'mask' in batch[0]:
        mask = torch.stack([d['mask'] for d in batch])
    if 'inpainting_mask' in batch[0]:
        inpainting_mask = torch.stack([d['inpainting_mask'] for d in batch])
    if 'text' in batch[0]:
        text = [d['text'] for d in batch]
    if 'meta' in batch[0]:
        meta = [d['meta'] for d in batch]
    return {
        'image': images,
        'panseg': semseg,
        'meta': meta,
        'text': text,
        'tokens': tokens,
        'mask': mask,
        'inpainting_mask': inpainting_mask,
        'image_panseg': image_panseg,
        'image_semseg': image_semseg,
        'gt_semseg': gt_semseg
    }

def pr_train_dataloader(p):
    transforms = get_train_transforms(p.transformation)
    train_dataset = get_dataset(
        split='train',
        db_name=p.db,
        transform=transforms,
        cfg_palette=p.pa
    )

    train_dataloader = DataLoader(
    train_dataset,
    batch_size=p.train.batch_size,
    num_workers=p.train.num_workers,
    shuffle=True,  #
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn,
    )

    return train_dataloader

def pr_val_dataloader(p):
    transforms_val = get_val_transforms(p.transformation)
    val_dataset = get_dataset(
        split='val',
        db_name=p.db,
        transform=transforms_val,
        cfg_palette=p.pa
    )

    val_dataloader = DataLoader(
    val_dataset,
    batch_size=p.eval.batch_size,
    num_workers=p.eval.num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_fn,
    )

    return val_dataloader

