import torch
from tqdm import tqdm

def inversion_forward_process(model, scheduler, x0, encoder_hidden_states, etas=None, prog_bar=False, cfg_scale=3.5, num_inference_steps=50):
    timesteps = scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        x0.shape[0],  # batch size from x0
        model.in_channels, 
        model.sample_size,
        model.sample_size
    )
    
    if etas is None:
        raise ValueError("etas must be provided")

    if isinstance(etas, (int, float)):
        etas = [etas] * num_inference_steps
        
    xts = sample_xts_from_x0(scheduler, x0, num_inference_steps=num_inference_steps)
    alpha_bar = scheduler.alphas_cumprod
    zs = torch.zeros(size=variance_noise_shape, device=model.device)  # batch dimension included
    
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps

    uncond_embedding = torch.zeros_like(encoder_hidden_states)
    #uncond_embedding = torch.randn(encoder_hidden_states.shape, device=model.device)
    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        xt = xts[idx + 1] 
        
        with torch.no_grad():
            cond_out = model.forward(xt, timestep=t, encoder_hidden_states=encoder_hidden_states)
            out = model.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
        
        noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        
        xtm1 = xts[idx][None]  # Ensure that xtm1 has the correct batch size
        pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
        prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        variance = get_variance(scheduler, t)
        pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5 * noise_pred
        mu_xt = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
        
        # Make sure the batch size is handled correctly when assigning to zs
        zs[idx] = z  # `zs[idx]` has the correct shape [8, 4, 32, 32], matching `z`
        
        xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
        xts[idx] = xtm1  # xts[idx] should have the correct batch shape

    zs[0] = torch.zeros_like(zs[0])  # Make the first timestep's noise zero
    
    return xt, zs, xts

def sample_xts_from_x0(scheduler, x0, num_inference_steps=50):
    # Scheduler precomputation
    alpha_bar = scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    timesteps = scheduler.timesteps.to(x0.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    
    # Create xts with the shape (num_inference_steps + 1, batch_size, channels, height, width)
    xts = torch.zeros((num_inference_steps + 1, *x0.shape)).to(x0.device)
    
    # Assign x0 to the first timestep in xts
    xts[0] = x0
    
    # Iterating over timesteps in reverse order
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    
    return xts


def get_variance(scheduler, timestep):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def reverse_step(scheduler, model_output, timestep, sample, eta=0, variance_noise=None):
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    variance = get_variance(scheduler, timestep)
    std_dev_t = eta * variance ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model_output.device)
        sigma_z = eta * variance ** 0.5 * variance_noise
        prev_sample = prev_sample + sigma_z
    return prev_sample

def inversion_reverse_process(model, scheduler, xT, etas=0, encoder_hidden_states=None, cfg_scales=None, prog_bar=False, zs=None):
    batch_size = encoder_hidden_states.shape[0]
    uncond_embedding = torch.zeros_like(encoder_hidden_states)
    #uncond_embedding = torch.randn(encoder_hidden_states.shape, device=model.device)
    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)
    if isinstance(etas, (int, float)):
        etas = [etas] * scheduler.num_inference_steps
    timesteps = scheduler.timesteps.to(model.device)
    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
    for t in op:
        idx = scheduler.num_inference_steps - t_to_idx[int(t)] - (scheduler.num_inference_steps - zs.shape[0] + 1)
        with torch.no_grad():
            cond_out = model.forward(xt, timestep=t, encoder_hidden_states=encoder_hidden_states)
            out = model.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
        
        noise_pred = out.sample + cfg_scales_tensor * (cond_out.sample - out.sample)
        xt = reverse_step(scheduler, noise_pred, t, xt, eta=etas[idx], variance_noise=zs[idx] if zs is not None else None)
    return xt, zs