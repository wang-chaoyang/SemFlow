import os
from accelerate import Accelerator,DistributedType
import shutil
import torch
import warnings

def resume_state(accelerator:Accelerator,args,num_update_steps_per_epoch,model):
    first_epoch = 0
    resume_step = 0
    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.env.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.env.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.env.gradient_accumulation_steps)
            accelerator.load_state(os.path.join(args.env.output_dir, path))
    return first_epoch, resume_step, global_step


def save_normal(accelerator:Accelerator,args,logger,global_step,model):
    accelerator.wait_for_everyone()
    args.env.checkpoints_total_limit = 1
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(args.env.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
      
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.env.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.env.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) > args.env.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.env.checkpoints_total_limit
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.env.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
    
    
