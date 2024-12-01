import sys
sys.path.append('.')
import os
import os.path as osp
import numpy as np
import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel
from einops import rearrange
from PIL import Image
import argparse
import imageio
from module.data.builder import build_palette
from module.data.prepare_text import sd_null_condition
from module.pipe.val import l2i
from module.pipe.pipe import pipeline_rf,pipeline_rf_reverse


@torch.no_grad()
def valrf(data,args,vae,unet,device,weight_dtype,null_condition):
    num_inference_steps = args.valstep
    guidance_scale = 1.0    
    palette = build_palette(6,50)
    cls_num = 19
    table = torch.tensor(palette[:cls_num*3])
    table = rearrange(table,'(q c) -> c q',c=3)
    table = table.to(device=device,dtype=weight_dtype)[None,:,:,None,None]  
    table = table / 127.5 -1.0
    prompt_embeds = null_condition
    unet_added_conditions = None
    timesteps = torch.arange(1,1000,1000//num_inference_steps).to(device=device).long()
    timesteps = timesteps.reshape(len(timesteps),-1).flip([0,1]).squeeze(1)

    rgb_images = data['image'].to(device=device, dtype=weight_dtype)
    rgb_latents = vae.encode(rgb_images).latent_dist.mode()*vae.scaling_factor
    bsz = rgb_latents.shape[0]
    encoder_hidden_states = prompt_embeds.repeat(bsz,1,1)  

    assert unet_added_conditions is None
    _unet_added_conditions = None
 
    if args.valmode=='seg':
        _, all_latents = pipeline_rf(timesteps,unet,rgb_latents,encoder_hidden_states,prompt_embeds,guidance_scale,_unet_added_conditions)
        imgs,_ = l2i(torch.concat(all_latents,dim=0),vae,weight_dtype)
        with imageio.get_writer(osp.join(args.sv_dir,'output_seg.gif'), mode='I', fps=4) as writer:
            for im in imgs:
                writer.append_data(im[:,:,::-1])
    elif args.valmode=='gen':
        image_semseg = data['image_semseg'].to(device=device, dtype=weight_dtype)
        noise = torch.rand_like(image_semseg)-0.5  
        image_semseg += noise*0.1
        image_latents = vae.encode(image_semseg).latent_dist.mode()*vae.scaling_factor
        _, all_latents = pipeline_rf_reverse(timesteps,unet,image_latents,encoder_hidden_states,prompt_embeds,guidance_scale,_unet_added_conditions)
        imgs,_ = l2i(torch.concat(all_latents,dim=0),vae,weight_dtype)
        with imageio.get_writer(osp.join(args.sv_dir,'output_gen.gif'), mode='I', fps=4) as writer:
            for im in imgs:
                writer.append_data(im[:,:,::-1])
    else:
        raise ValueError()




def main(args):

    device = args.device
    if args.seed is not None:
        set_seed(args.seed)
    vae = AutoencoderKL.from_pretrained(args.pretrain_model, subfolder='vae', revision=None)
    unet = UNet2DConditionModel.from_pretrained(args.ckpt)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
   
    vae.requires_grad_(False)
    unet.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)

    null_condition = sd_null_condition(args.pretrain_model)
    null_condition = null_condition.to(device=device, dtype=weight_dtype)
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    torch.cuda.empty_cache()
    data = prepare_data(args.im_path,args.mask_path)
    unet.eval()
    valrf(data,args,vae,unet,device,weight_dtype,null_condition)
    
def prepare_pm(x):
    palette = build_palette(6,50)
    assert len(x.shape)==2
    h,w = x.shape
    pm = np.ones((h,w,3))*255
    clslist = np.unique(x).tolist()
    if 255 in clslist:
        raise ValueError()
    for _c in clslist:
        _x,_y = np.where(x==_c)
        pm[_x,_y,:] = palette[int(_c)*3:(int(_c)+1)*3]
    return pm

def prepare_data(ipth,mpth):
    import torchvision.transforms as T
    image = Image.open(ipth).convert('RGB').resize((512,512),resample=Image.Resampling.BILINEAR)
    seg = Image.open(mpth).resize((512,512),resample=Image.Resampling.NEAREST)
    image_semseg = prepare_pm(np.array(seg)).astype(np.uint8)
    tf = T.Compose([T.ToTensor(),T.Normalize(0.5,0.5)])
    image = tf(image)
    image_semseg = tf(image_semseg)
    data = dict(image=image.unsqueeze(0),image_semseg=image_semseg.unsqueeze(0))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model',type=str,default='dataset/pretrain/stable-diffusion-v1-5')
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--allow_tf32',type=bool,default=True)
    parser.add_argument('--mixed_precision',type=str,default=None)
    parser.add_argument('--valmode',type=str,default='gen',choices=['seg','gen'])
    parser.add_argument('--valstep',type=int,default=25)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--ckpt',type=str,default='demo/unet')
    parser.add_argument('--sv_dir',type=str,default='demo/vis')
    parser.add_argument('--im_path',type=str,default='file/img.jpg')
    parser.add_argument('--mask_path',type=str,default='file/mask.png')
    args = parser.parse_args()
    os.makedirs(args.sv_dir,exist_ok=True)
    main(args)