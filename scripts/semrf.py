import sys
sys.path.append('.')

import logging
import math
import os

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DeepSpeedPlugin

from tqdm import tqdm
from omegaconf import OmegaConf
import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from module.data.load_dataset import pr_val_dataloader, pr_train_dataloader
from module.pipe.val import valrf
from module.data.hook import resume_state, save_normal
from module.data.prepare_text import sd_null_condition

logger = get_logger(__name__)


@torch.no_grad()
def pre_rf(data, vae, device, weight_dtype, args):
    rgb_images = data['image'].to(dtype=weight_dtype, device=device)
    images = data['image_semseg'].to(dtype=weight_dtype, device=device)
    noise = torch.rand_like(images)-0.5
    images = images + args.pert.co*noise
    rgb_latents = vae.encode(rgb_images).latent_dist.sample()* vae.scaling_factor

    latents = vae.encode(images).latent_dist.mode()* vae.scaling_factor

    return latents, rgb_latents


def main(args):

    ttlsteps = 1000
    args.transformation.size = args.env.size

    print('init SD 1.5')
    args.pretrain_model = 'dataset/pretrain/stable-diffusion-v1-5'
    args.vae_path = 'dataset/pretrain/stable-diffusion-v1-5/vae'

    train_dataloader = pr_train_dataloader(args)
    val_dataloader = pr_val_dataloader(args)
   
    logging_dir = Path(args.env.output_dir, args.env.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.env.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.env.gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.env.gradient_accumulation_steps,
        mixed_precision=args.env.mixed_precision,
        log_with=args.env.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin if args.env.deepspeed else None
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.env.seed is not None:
        set_seed(args.env.seed)


    if accelerator.is_main_process:
        if args.env.output_dir is not None:
            os.makedirs(os.path.join(args.env.output_dir,'vis'), exist_ok=True)
            OmegaConf.save(args,os.path.join(args.env.output_dir,'config.yaml'))

    vae = AutoencoderKL.from_pretrained(args.vae_path, revision=None)
    unet = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet", revision=None)
    
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
   
    vae.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    null_condition = sd_null_condition(args.pretrain_model)
    null_condition = null_condition.to(accelerator.device, dtype=weight_dtype)
    
    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.env.scale_lr:
        args.optim.lr = (
            args.optim.lr * args.env.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    assert args.optim.name=='adamw'
    optimizer_class = torch.optim.AdamW  
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.optim.lr,
        betas=(args.optim.beta1, args.optim.beta2),
        weight_decay=args.optim.weight_decay,
        eps=args.optim.epsilon,
    )

    # # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    assert args.env.max_train_steps is not None

    lr_ratio = 1 if args.env.deepspeed else accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_scheduler.warmup_steps * lr_ratio,
        num_training_steps=args.env.max_train_steps * lr_ratio,
    )

    unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, val_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.env.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.env.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.env.max_train_steps / num_update_steps_per_epoch)

    ##
    if accelerator.is_main_process:
        accelerator.init_trackers("model")

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.env.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.env.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.env.max_train_steps}")
    global_step = 0
    first_epoch = 0

    first_epoch, resume_step, global_step = resume_state(accelerator,args,num_update_steps_per_epoch,unet)
    torch.cuda.empty_cache()
    progress_bar = tqdm(range(global_step, args.env.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    device = accelerator.device

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            unet.train()
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.env.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                
                latents, z0 = pre_rf(batch,vae,device,weight_dtype,args)
                bsz = latents.shape[0]
                if args.cfg.continus:
                    t = torch.rand((bsz,),device=device,dtype=weight_dtype)
                    timesteps = t*ttlsteps
                else:
                    timesteps = torch.randint(0, ttlsteps,(bsz,), device=device,dtype=torch.long)
                    t = timesteps.to(weight_dtype)/ttlsteps

                t = t[:,None,None,None]
                perturb_latent = t*latents+(1.-t)*z0

                prompt_embeds = null_condition.repeat(bsz,1,1)
                model_pred = unet(perturb_latent, timesteps, prompt_embeds).sample
                target = latents - z0

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train.batch_size)).mean()
                train_loss += avg_loss.item() / args.env.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.env.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.env.checkpointing_steps == 0:
                    save_normal(accelerator,args,logger,global_step,unet)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if args.env.val_iter > 0 and global_step % args.env.val_iter == 0:
                unet.eval()
                
                valrf(
                    accelerator,
                    args,
                    vae,
                    unet,
                    val_dataloader,
                    device,
                    weight_dtype,
                    null_condition,
                    max_iter=None,
                    gstep=global_step
                    )

            if global_step >= args.env.max_train_steps:
                accelerator.wait_for_everyone()
                break

    accelerator.end_training()


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    assert os.path.isfile(cfg_path)
    args = OmegaConf.load(cfg_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args,cli_config)
    main(args)
