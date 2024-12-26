import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from core.base_network import BaseNetwork
from .net.UNet import RestorationNet

''' 
    NoiseDA serves as a general DA training strategy for image restoration tasks, 
    it can be flexibly applied to various image restoration networks as follows.
'''
#from .net.dncnn import DnCNN as RestorationNet
#from .net.uformer import Uformer as RestorationNet
#from .net.swinir import SwinIR as RestorationNet
#from .net.restormer import Restormer as RestorationNet

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.RestorationNet = RestorationNet()
        
    def set_loss(self, loss_fn):
        self.mse_loss = loss_fn[0]
        self.pix_loss = loss_fn[1]
        self.triplet_margin_loss = loss_fn[-1]

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1]) 

        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_syn=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_syn, y_t, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_syn=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_syn=y_syn)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration_test_head(self, input_image):
        input_image_residual = self.RestorationNet(input_image)
        input_image_correct = input_image + input_image_residual
        return input_image_correct

    def forward(self, y_0, y_syn=None, y_real=None, ref_img=None, diff_flag=0, noise=None):
        '''
        diff_flag settings:
        0: vanilla restoration net
        1: restoration net + diffusion model (shortcut eliminated, aka Ours)
        2: restoration net + diffusion model (unpaired clean large-scale image dataset, aka Ours_Ex)
        
        input parsing:
        y_syn: the synthetic restored image
        y_0: the ground truth of the synthetic restored image
        y_real: the real-world restored image
        ref_img: the unpaired clean image from other datasets
        '''
        
        if(diff_flag==0):
            y_syn_residual = self.RestorationNet(y_syn)
            y_syn_correct = y_syn + y_syn_residual
            loss_syn = self.pix_loss(y_0, y_syn_correct)
            return loss_syn
        
        if(diff_flag==1):
            #noise schedule
            b, *_ = y_0.shape
            t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
            gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1)

            noise = default(noise, lambda: torch.randn_like(y_0))
            y_noisy = self.q_sample(                                         
                y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
            
            #image restoration on synthetic and real-world data
            y_syn_residual = self.RestorationNet(y_syn)
            y_real_residual = self.RestorationNet(y_real)
            y_syn_correct = y_syn + y_syn_residual
            y_real_correct = y_real + y_real_residual
            
            #randomly swap/shuffle the synthetic and real data in each batch
            mask = torch.rand(b) > 0.5
            y_syn_correct_swap = y_syn_correct.clone()
            y_real_correct_swap = y_real_correct.clone()
            y_syn_correct_swap[mask] = y_real_correct[mask]
            y_real_correct_swap[mask] = y_syn_correct[mask]
            
            #swap the residual to formulate the negative sample
            y_syn_correct_neg = y_syn + y_real_residual
            y_real_correct_neg = y_real + y_syn_residual
            mask_neg = torch.rand(b) > 0.5
            y_syn_correct_neg_swap = y_syn_correct_neg.clone()
            y_real_correct_neg_swap = y_real_correct_neg.clone()
            y_syn_correct_neg_swap[mask_neg] = y_real_correct_neg[mask_neg]
            y_real_correct_neg_swap[mask_neg] = y_syn_correct_neg[mask_neg]
            
            #diffusion denoising process
            noise_hat = self.denoise_fn(torch.cat([y_noisy, y_syn_correct_swap, y_real_correct_swap], dim=1), sample_gammas)
            noise_hat_neg = self.denoise_fn(torch.cat([y_noisy, y_syn_correct_neg_swap, y_real_correct_neg_swap], dim=1), sample_gammas)
            
            #loss computation
            loss_syn = self.pix_loss(y_0, y_syn_correct)
            loss_noise = (self.mse_loss(noise, noise_hat) + self.triplet_margin_loss(noise, noise_hat, noise_hat_neg)) / 2.

            return loss_noise, loss_syn
        
        if(diff_flag==2):
            #noise schedule
            b, *_ = y_0.shape
            t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
            gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
            sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
            sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
            sample_gammas = sample_gammas.view(b, -1)
            
            # use the unpared clean image from other dataset as the diffusion input
            noise = default(noise, lambda: torch.randn_like(y_0))
            y_noisy = self.q_sample(                                         
                y_0=ref_img, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
            
            #image restoration on synthetic and real data
            y_syn_residual = self.RestorationNet(y_syn)
            y_real_residual = self.RestorationNet(y_real)
            y_syn_correct = y_syn + y_syn_residual
            y_real_correct = y_real + y_real_residual
            
            #randomly swap/shuffle the synthetic and real data in each batch
            mask = torch.rand(b) > 0.5
            y_syn_correct_swap = y_syn_correct.clone()
            y_real_correct_swap = y_real_correct.clone()
            y_syn_correct_swap[mask] = y_real_correct[mask]
            y_real_correct_swap[mask] = y_syn_correct[mask]
            
            #diffusion denoising process
            noise_hat = self.denoise_fn(torch.cat([y_noisy, y_syn_correct_swap, y_real_correct_swap], dim=1), sample_gammas)
            loss_noise = self.mse_loss(noise, noise_hat)
            loss_syn = self.pix_loss(y_0, y_syn_correct)

            return loss_noise, loss_syn
        
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas