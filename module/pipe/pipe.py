import torch

@torch.no_grad()
def pipeline_rf(timesteps,unet,z0,encoder_hidden_states,blank_feat,guidance_scale,unet_added_conditions=None):
    cls_free = guidance_scale>1.0
    bsz = z0.shape[0]
    if cls_free:
        blank_feat = blank_feat.repeat(bsz,1,1)
        encoder_hidden_states = torch.cat([encoder_hidden_states, blank_feat], dim=0)
    ttlsteps = len(timesteps)
    timesteps = timesteps.reshape(ttlsteps,-1).flip([0,1]).squeeze(1)-1
    dt = 1.0 / ttlsteps
    latents = z0
    all_latents = []
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if cls_free else latents
        v_pred = unet(latent_model_input, t, encoder_hidden_states, added_cond_kwargs=unet_added_conditions, return_dict=False)[0]
        if cls_free:
            v_pred_text, v_pred_null = v_pred.chunk(2)
            v_pred = v_pred_null + guidance_scale * (v_pred_text - v_pred_null)
        latents = latents + dt * v_pred 
        all_latents.append(latents)
    return latents, all_latents


@torch.no_grad()
def pipeline_rf_reverse(timesteps,unet,z1,encoder_hidden_states,blank_feat,guidance_scale,unet_added_conditions=None):
    cls_free = guidance_scale>1.0
    bsz = z1.shape[0]
    if cls_free:
        blank_feat = blank_feat.repeat(bsz,1,1)
        encoder_hidden_states = torch.cat([encoder_hidden_states, blank_feat], dim=0)
    ttlsteps = len(timesteps)
    timesteps = 1000 - timesteps.max()+timesteps
    dt = 1.0 / ttlsteps
    latents = z1
    all_latents = []
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if cls_free else latents
        v_pred = unet(latent_model_input, t, encoder_hidden_states, added_cond_kwargs=unet_added_conditions, return_dict=False)[0]
        if cls_free:
            v_pred_text, v_pred_null = v_pred.chunk(2)
            v_pred = v_pred_null + guidance_scale * (v_pred_text - v_pred_null)
        latents = latents - dt * v_pred 
        all_latents.append(latents)
    return latents, all_latents