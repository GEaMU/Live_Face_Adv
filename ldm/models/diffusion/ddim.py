"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import torch.nn.functional as F
from torchvision import transforms

# 定义图像resize工具（适配人脸识别模型输入，如112x112）
resize = transforms.Resize((112, 112))

def normalize_feature(feat):
    """L2归一化特征（人脸识别必须步骤）"""
    return F.normalize(feat, p=2, dim=1)

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.device = device

        # 特征驱动扰动的核心参数（从kwargs读取）
        self.pgd_eps = kwargs.get('pgd_eps', 128 / 255)  # 最大扰动幅度（全局）
        self.pgd_alpha_base = kwargs.get('pgd_alpha', 128 / 255)  # 基础步长
        self.target_classifier = kwargs.get('classifier', None)  # 人脸识别模型
        self.original_images = kwargs.get('source_img', None)  # 原始源图像
        self.target_images = kwargs.get('tgt_image', None)  # 目标图像
        self.masked = kwargs.get('mask', None)  # 目标图像

        # 扩散时间步相关参数（动态调整用）
        self.ddim_alphas = None
        self.ddim_alphas_prev = None
        self.ddim_sigmas = None
        self.ddim_steps = None
        self.current_t = None  # 记录当前扩散时间步
        self.total_ddim_steps = None  # 总DDIM采样步数

        # 保存累计扰动（用于跨时间步的扰动约束）
        self.delta_accum = None


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != self.device:
                attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, _t=1000):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=_t, verbose=verbose)
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

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def _get_dynamic_alpha(self):
        """
        根据扩散时间步动态调整扰动步长：
        - 早期时间步（t大，噪声多）：步长大，快速调整特征
        - 晚期时间步（t小，图像清晰）：步长小，精细调整，避免破坏图像
        """
        # 归一化当前时间步到[0,1]（0=最早步，1=最晚步）
        t_norm = 1.0 - (self.current_t / self.ddpm_num_timesteps)
        # 步长随时间步衰减（指数衰减，可调整系数）
        dynamic_alpha =1# self.pgd_alpha_base * (0.1 + 0.9 * torch.exp(-5 * t_norm))
        return dynamic_alpha

    def _get_loss_weights(self):
        """
        根据扩散时间步动态调整损失权重：
        - 早期：重点抑制原特征（先脱离原身份）
        - 晚期：重点拟合目标特征（精准对齐目标身份）
        """
        t_norm = self.current_t / self.ddpm_num_timesteps  # 0=晚期，1=早期
        # 目标特征损失权重：晚期增大
        origin_weight = 5.0 + 5.0 * (1 - t_norm)  # 5~10
        # 原特征抑制权重：早期增大
        target_weight = 1.0 + 4.0 * t_norm  # 1~5
        return target_weight, origin_weight

    def feature_driven_gradient_step(
            self,
            images,  # 当前扩散时间步的图像
            diffusion_model,
            device,
            mask
    ):
        """
        单步梯度扰动（替代多步PGD）：
        1. 跟随扩散时间步动态调整步长、损失权重
        2. 累计扰动并约束在epsilon范围内
        3. 保留特征驱动的核心逻辑（拟合目标+抑制原特征）
        """
        # 校验必要参数（缺失则返回原图像）
        if self.target_classifier is None or self.original_images is None or self.target_images is None:
            return images

        # 初始化累计扰动（首次调用时）
        if self.delta_accum is None:
            self.delta_accum = torch.zeros_like(images, device=device)

        # ------------------------------
        # 预提取目标/原图像的特征（固定，无梯度）
        # ------------------------------
        target_img_resized = resize(self.target_images).to(device)
        original_img_resized = resize(self.original_images).to(device)

        with torch.no_grad():
            target_feat = self.target_classifier(target_img_resized)
            target_feat = normalize_feature(target_feat)
            original_feat = self.target_classifier(original_img_resized)
            original_feat = normalize_feature(original_feat)

        # ------------------------------
        # 单步梯度计算（核心：仅1次反向传播）
        # ------------------------------
        images.requires_grad = True  # 开启梯度
        self.target_classifier.eval()
        diffusion_model.eval()

        # 提取当前图像的特征（保留梯度）
        img_resized = resize(images)
        current_feat = self.target_classifier(img_resized)
        current_feat = normalize_feature(current_feat)

        # 动态获取损失权重（随时间步变化）
        target_weight, origin_weight = self._get_loss_weights()

        # ------------------------------
        # 核心损失：动态权重的特征拟合+原特征抑制
        # ------------------------------
        # 损失1：拟合目标特征（MSE+余弦相似度，动态权重）
        loss_recon_mse = F.mse_loss(current_feat, target_feat)
        loss_recon_cos = 1 - F.cosine_similarity(current_feat, target_feat, dim=1).mean()
        loss_target = target_weight * (loss_recon_mse + loss_recon_cos)

        # 损失2：抑制原特征（动态权重）
        loss_origin_suppress = - origin_weight * (1 - F.cosine_similarity(current_feat, original_feat, dim=1).mean())
        total_loss = loss_target + 100*loss_origin_suppress

        # ------------------------------
        # 梯度计算与单步更新
        # ------------------------------
        # 清空旧梯度
        if images.grad is not None:
            images.grad.zero_()
        # 反向传播
        total_loss.backward()

        # 动态获取当前步的扰动步长
        dynamic_alpha = self._get_dynamic_alpha()
        # 梯度符号更新（保留PGD的符号特性，更稳定）
        grad_sign = images.grad.sign()

        if mask is not None:
            # mask=1是人脸区域，mask=0是非人脸区域
            grad = grad_sign * (1. - self.masked) * 3.0 + grad_sign * self.masked * 0.01  # 非人脸梯度×3，人脸×1
        grad_sign = grad.sign()

        # ------------------------------
        # 累计扰动并约束（关键：全局扰动不超过epsilon）
        # ------------------------------
        # 累加单步扰动
        self.delta_accum += dynamic_alpha * grad_sign
        # 约束累计扰动在[-epsilon, epsilon]范围内
        self.delta_accum = torch.clamp(self.delta_accum, min=-self.pgd_eps, max=self.pgd_eps)
        # 应用扰动到原图像
        images_adv = images.detach() + self.delta_accum
        # 像素值约束（扩散模型输出范围[-1, 1]）
        images_adv = torch.clamp(images_adv, min=-1, max=1)

        # ------------------------------
        # 日志输出（可选：监控特征相似度）
        # ------------------------------
        with torch.no_grad():
            sim_target = F.cosine_similarity(current_feat, target_feat, dim=1).mean().item()
            sim_origin = F.cosine_similarity(current_feat, original_feat, dim=1).mean().item()
            print(
                f"Diffusion Step {self.current_t} | Target Sim: {sim_target:.4f} | Origin Sim: {sim_origin:.4f} | Alpha: {dynamic_alpha:.6f}")

        return images_adv.detach()

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
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
               _t=1000,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               # 特征驱动扰动的参数
               pgd_eps=128 / 255,
               pgd_alpha=128 / 255,
               target_classifier=None,
               original_images=None,
               target_images=None,
               **kwargs
               ):
        # 重置累计扰动（每次采样前清空）
        self.delta_accum = None
        # 配置扰动参数
        self.pgd_eps = pgd_eps
        self.pgd_alpha_base = pgd_alpha
        self.target_classifier = target_classifier
        self.original_images = original_images
        self.target_images = target_images

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

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, _t=_t)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
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
                                                    ucg_schedule=ucg_schedule,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?

                # ========== 核心修复1：正确的维度/尺寸对齐 ==========
                # 1. 将img_orig和mask缩放到img的尺寸（64x64）
                img_orig = F.interpolate(img_orig, size=img.shape[-2:], mode='bilinear', align_corners=False)
                mask_resized = F.interpolate(mask, size=img.shape[-2:], mode='nearest')  # 重命名避免覆盖原mask

                # 2. 二值化mask（确保只有0/1，无中间值）
                mask_resized = (mask_resized > 0.5).float()

                # 3. 统一通道数到img的通道数（关键：之前repeat(1,1,1,1)未扩展通道）
                mask_resized = mask_resized.repeat(1, img.shape[1], 1, 1)  # [1,1,64,64] → [1,4,64,64]
                img_orig = F.pad(img_orig, (0, 0, 0, 0, 0, img.shape[1] - img_orig.shape[1]))  # 补通道到4

                # 4. 强制设备/类型对齐
                img_orig = img_orig.to(device=img.device, dtype=img.dtype)
                mask_resized = mask_resized.to(device=img.device, dtype=img.dtype)

                # 5. 执行混合（此时mask_resized是严格二值，通道/尺寸完全匹配）
                img = img_orig * mask_resized + (1. - mask_resized) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,
                                      **kwargs)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,mask=None, **kwargs):
        b, *_, device = *x.shape, x.device

        # 记录当前扩散时间步（用于动态调整）
        self.current_t = t.item() if isinstance(t, torch.Tensor) else t

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

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # ------------------------------
        # 核心优化：单步特征驱动梯度扰动（替代多步PGD）
        # ------------------------------
        if self.target_classifier is not None:
            # 临时启用梯度（因为扰动需要反向传播）
            self.model.eval()
            with torch.enable_grad():
                x_adv = self.feature_driven_gradient_step(
                    images=x,
                    diffusion_model=self.model,
                    device=device,
                    mask=mask
                )
        else:
            x_adv = x

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x_adv - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x_adv, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # classifier guidance
        x_target = kwargs['x_target']
        classifier = kwargs['classifier']
        classifier_scale = kwargs['classifier_scale']
        x_or = kwargs['source_img']
        grad = self.get_classifier_guidance(pred_x0, x_target, classifier, classifier_scale,x_or)

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise + sigma_t * grad.float()
        # x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise + grad.float()
        return x_prev, pred_x0

    def get_classifier_guidance(self, x_0, x_target, classifier, classifier_scale,x_or):
        from torch.nn import functional as F
        with torch.enable_grad():
            B = x_0.shape[0]
            score = 0
            i = 0
            x_in = x_0.detach().requires_grad_(True)

            img = self.model.decode_first_stage(x_in)
            # from torchvision.utils import save_image
            for k, v in classifier.items():
                resize = torch.nn.AdaptiveAvgPool2d((112, 112)) if k != 'FaceNet' \
                    else torch.nn.AdaptiveAvgPool2d((160, 160))

                feature1 = v(resize(img)).reshape(B, -1)
                feature2 = v(resize(x_target)).reshape(B, -1)
                feature3 = v(resize(x_or)).reshape(B, -1)

                score += (F.cosine_similarity(feature1, feature2).sum() / B)#-(F.cosine_similarity(feature1, feature3).sum() / B))
                i += 1

            # print(score.item() / i)
            return torch.autograd.grad(score / i, x_in)[0] * classifier_scale#*10
    # def get_classifier_guidance(self, x_0, x_target, classifier, classifier_scale):
    #     from torch.nn import functional as F
    #     with torch.enable_grad():
    #         B = x_0.shape[0]
    #         score = 0
    #         i = 0
    #         x_in = x_0.detach().requires_grad_(True)
    #
    #         img = self.model.decode_first_stage(x_in)
    #         # from torchvision.utils import save_image
    #         for k, v in classifier.items():
    #             resize = torch.nn.AdaptiveAvgPool2d((112, 112)) if k != 'FaceNet' \
    #                 else torch.nn.AdaptiveAvgPool2d((160, 160))
    #
    #             feature1 = v(resize(img)).reshape(B, -1)
    #             feature2 = v(resize(x_target)).reshape(B, -1)
    #
    #             score += F.cosine_similarity(feature1, feature2).sum() / B
    #             i += 1
    #
    #         # print(score.item() / i)
    #         return torch.autograd.grad(score / i, x_in)[0] * classifier_scale#*10


    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec