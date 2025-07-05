"""SAMPLING ONLY."""

import torch
import numpy as np
from einops import rearrange

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters_dydiff, make_ddim_timesteps, noise_like, extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        thetas_cumprod = self.model.thetas_cumprod
        self.register_buffer('gammas', to_torch(self.model.gammas))
        self.register_buffer('thetas_cumprod', to_torch(thetas_cumprod))
        self.register_buffer('thetas_cumprod_prev', to_torch(self.model.thetas_cumprod_prev))

        self.register_buffer('one_minus_thetas_cumprod', to_torch(1. - thetas_cumprod.cpu()))
        self.register_buffer('recip_thetas_cumprod', to_torch(1. / thetas_cumprod.cpu()))
        self.register_buffer('recipm1_thetas_cumprod', to_torch(1. / thetas_cumprod.cpu() - 1))
        self.register_buffer('recip_thetas_cumprod_sqrt_recip_alphas_cumprod', to_torch(1. / thetas_cumprod.cpu() * np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('recip_thetas_cumprod_sqrt_recipm1_alphas_cumprod', to_torch(1. / thetas_cumprod.cpu() * np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev, ddim_thetas, ddim_thetas_prev = make_ddim_sampling_parameters_dydiff(
            alphacums=alphas_cumprod.cpu(),
            thetacums=thetas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,verbose=verbose
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_thetas', ddim_thetas)
        self.register_buffer('ddim_thetas_prev', ddim_thetas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - (self.alphas_cumprod / self.alphas_cumprod_prev) * (self.thetas_cumprod / self.thetas_cumprod_prev) ** 2 - self.alphas_cumprod * (1 - (self.thetas_cumprod / self.thetas_cumprod_prev) ** 2)))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               x_prev,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(x_prev, conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, x_prev, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        if x_T is None:
            b, tc, h, w = shape
            img = torch.randn(b, tc, h, w, device=device)
            thetas = self.model.thetas_cumprod if ddim_use_original_steps else self.ddim_thetas
            # theta_t = torch.full((b, 1, 1, 1), thetas[timesteps[-1]], device=device)
            theta_t = torch.full((b, 1, 1, 1), thetas[-1], device=device)
            if self.model.use_x_ema != 'only':
                img = self.model.get_emas(img, theta_t.sqrt(), (1. - theta_t).sqrt())
        
        else:
            img = x_T
        
        if self.model.ensemble:
            img = torch.cat([img, img], dim=1)

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = time_range

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, x_prev, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        if self.model.ensemble:
            ###### MODIFY HERE ######
            _, img = img.chunk(2, dim=1)
            # img, _ = img.chunk(2, dim=1)
            ###### MODIFY HERE ######

            # for idx in range(len(intermediates["x_inter"])):
            #     _, intermediates["x_inter"][idx] = intermediates["x_inter"][idx].chunk(2, dim=1)
            # for idx in range(len(intermediates["pred_x0"])):
            #     _, intermediates["pred_x0"][idx] = intermediates["pred_x0"][idx].chunk(2, dim=1)
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, x_prev, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.ensemble:
            x, x_raw = x.chunk(2, dim=1)
            if self.model.parameterization == "v_standard":
                e_t_raw = self.model.raw_predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t_raw = model_output
        
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, x_prev, t, model_output)
        elif self.model.parameterization == "v_standard":
            e_t = self.model.predict_eps_from_z_and_v_standard(x, x_prev, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        thetas = self.model.thetas_cumprod if use_original_steps else self.ddim_thetas
        thetas_prev = self.model.thetas_cumprod_prev if use_original_steps else self.ddim_thetas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        theta_t = torch.full((b, 1, 1, 1), thetas[index], device=device)
        theta_prev = torch.full((b, 1, 1, 1), thetas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization == "eps":
            # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # pred_x0 = ((x - sqrt_one_minus_at * e_t) / a_t.sqrt() - (1. - theta_t) * x_prev) / theta_t
            pred_x0 = self.model.predict_start_from_noise(x, x_prev, t, model_output)
        elif self.model.parameterization == "v":
            pred_x0 = self.model.predict_start_from_z_and_v(x, x_prev, t, model_output)
        elif self.model.parameterization == "v_standard":
            pred_x0 = self.model.predict_start_from_z_and_v_standard(x, x_prev, t, model_output)
        else:
            pred_x0 = model_output
        
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if self.model.ensemble:
            if self.model.parameterization == "eps":
                pred_x0_raw = (x_raw - sqrt_one_minus_at * e_t) / a_t.sqrt()
            elif self.model.parameterization == "v_standard":
                pred_x0_raw = self.model.raw_predict_start_from_z_and_v(x_raw, t, model_output)
            else:
                raise NotImplementedError()
            if quantize_denoised:
                pred_x0_raw, _, *_ = self.model.first_stage_model.quantize(pred_x0_raw)


        if dynamic_threshold is not None:
            raise NotImplementedError

        # direction pointing to x_t
        if (sigma_t != 0).any():
            raise NotImplementedError
        
        if self.model.use_x_ema:
            pred_x0 = self.model.get_emas(pred_x0, theta_prev.sqrt(), (1. - theta_prev).sqrt())
        if self.model.use_x_ema != "only":
            e_t = self.model.get_reverse_emas(e_t, theta_t.sqrt(), (1. - theta_t).sqrt())
            e_t = self.model.get_emas(e_t, theta_prev.sqrt(), (1. - theta_prev).sqrt())
        x_last = a_prev.sqrt() * pred_x0 + (1. - a_prev).sqrt() * e_t

        return x_last, pred_x0
