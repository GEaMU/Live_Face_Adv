import torch
from torch import nn
from utils import setup_seed, get_fr_model, initialize_model, asr_calculation
import os
from FaceParsing.interface import FaceParsing
from dataset import base_dataset,threeD_dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import argparse
from torch.utils.data import Subset
# import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image

# import cv2
import imageio
import numpy as np
from tqdm import tqdm
from model import MattingBase, MattingRefine
def preprocess_nersemble( image,args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load("path/to/assets/BackgroundMattingV2/pytorch_resnet101.pth", map_location='cpu'), strict=False)
    src = (image.float() / 255.0).to(device, non_blocking=True)  # 假设输入是 [0, 255] 张量
    bgr = torch.ones_like(src) 
    with torch.no_grad():
        pha, _, _, _, _, _ = model(src, bgr) 
    mask = pha.repeat(1, 1, 1, 1)  # (batch, 3, H, W)
    print(mask.size())
    return mask.cpu()  





import argparse
import torch
import os


resize = transforms.Resize((112, 112))
    #transforms.Resize((112, 112))if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))


def normalize_feature(feat):
    return F.normalize(feat, p=2, dim=1)


def feature_driven_pgd_attack(
        diffusion_model,
        images,  
        original_images,  
        target_images, 
        attack_model, 
        epsilon,#=8 / 255,
        alpha=2 / 255, 
        iterations=2, 
        device="cuda"
):
    target_img_resized = resize(target_images).to(device)
    original_img_resized = resize(original_images).to(device)

    with torch.no_grad():
        target_feat = attack_model(target_img_resized)
        target_feat = normalize_feature(target_feat)
        original_feat = attack_model(original_img_resized)
        original_feat = normalize_feature(original_feat)
    images = images.to(device).clone()
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, min=-1, max=1) 
    images.requires_grad = True  

    attack_model.eval()
    diffusion_model.eval()

    for i in range(iterations):
        with torch.enable_grad():
            img_resized = resize(images)
            current_feat = attack_model(img_resized)
            current_feat = normalize_feature(current_feat)
            loss_recon_mse = F.mse_loss(current_feat, target_feat)
            loss_recon_cos = 1 - F.cosine_similarity(current_feat, target_feat, dim=1).mean()
            loss_target = 10 * (loss_recon_mse + loss_recon_cos)  # 权重放大10倍
            loss_origin_suppress = - (1 - F.cosine_similarity(current_feat, original_feat, dim=1).mean())
            total_loss = loss_target + loss_origin_suppress
        if images.grad is not None:
            images.grad.zero_()

        total_loss.backward()
        images.data = images.data - alpha * images.grad.sign() 
        delta = torch.clamp(images - images.detach().clone(), min=-epsilon, max=epsilon)
        images.data = images.detach().clone() + delta
        images.data = torch.clamp(images.data, min=-1, max=1)

        images.requires_grad = True
        sim_target = F.cosine_similarity(current_feat, target_feat, dim=1).mean().item()
        sim_origin = F.cosine_similarity(current_feat, original_feat, dim=1).mean().item()
        if (i + 1) % 4 == 0:  # 每4步打印一次
            print(f"Iteration {i + 1} | Target Sim: {sim_target:.4f} | Origin Sim: {sim_origin:.4f}")

    return images.detach()


@torch.no_grad()
def main(args):
    for sss in range(1):
        SEED=2025
        seed = SEED
        setup_seed(SEED)
        h = 512
        w = 512
        txt = ''
        ddim_steps = 45
        scale = 0
        classifier_scale = args.s
        batch_size = 1
        num_workers = 0

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        transform = transforms.Compose([transforms.Resize((512, 512)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        if args.dataset == 'celeba':
            dataset = base_dataset(dir='./celeba-hq_sample', transform=transform)
        elif args.dataset == 'ffhq':
            dataset = base_dataset(dir='./ffhq_sample', transform=transform)
        elif args.dataset == '3D':
            dataset = threeD_dataset(dir='./3Dnew', transform=transform)
        # dataset = base_dataset(dir='./s', transform=transform)
        dataset = Subset(dataset, [x for x in range(args.num)])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        sampler = initialize_model('configs/stable-diffusion/v2-inpainting-inference.yaml',
                                   'pretrained_model/512-inpainting-ema.ckpt')
        model = sampler.model

        # prng = np.random.RandomState(seed)
        # start_code = prng.randn(batch_size, 4, h // 8, w // 8)
        # start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

        # attack_model_names = ['IR152', 'IRSE50', 'FaceNet', 'MobileFace']
        # attack_model_names = ['IRSE50']
        attack_model_names = [args.model]
        attack_model_dict = {'IR152': get_fr_model('IR152'), 'IRSE50': get_fr_model('IRSE50'),
                             'FaceNet': get_fr_model('FaceNet'), 'MobileFace': get_fr_model('MobileFace')}
        # attack_model_resize_dict = {'IR152': 112, 'IRSE50': 112, 'FaceNet': 160, 'MobileFace': 112}
        # cos_sim_scores_dict = {'IR152': [], 'IRSE50': [], 'FaceNet': [], 'MobileFace': []}
        # cos_sim_scores_dict = {'IRSE50': []}
        cos_sim_scores_dict = {args.model: []}
        cos_sim_scores_dict_f2 = {args.model: []}
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for attack_model_name in attack_model_names:
            attack_model = attack_model_dict[attack_model_name]
            classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
            resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d(
                (160, 160))
            with torch.no_grad():
                save_num=0
                for i, (image, tgt_image) in enumerate(dataloader):
                    # print(image)
                    save_num+=1

                    img_or=image.clone()

                    tgt_image = tgt_image.to(device)
                    B = image.shape[0]
                    print(tgt_image.size())

                    face_parsing = FaceParsing()
                    pred = face_parsing(image)

                    # label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                    # 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
                    def get_mask(number):
                        return pred == number
                    # masks = [1, 2, 3, 4, 5, 6, 7,8,9, 10, 11, 12,13,16,17,18]#
                    masks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
                    # # # skin = get_mask(1)
                    # # # nose = get_mask(2)
                    # # # eye_gla = get_mask(3)
                    # # # l_eye = get_mask(4)
                    # # # r_eye = get_mask(5)
                    # # # l_brow = get_mask(6)
                    # # # r_brow = get_mask(7)
                    # # # mouth = get_mask(10)
                    # # # u_lip = get_mask(11)
                    # # # l_lip = get_mask(12)
                    mask = None
                    for x in masks:
                        if mask is not None:
                            mask |= get_mask(x)
                        else:
                            mask = get_mask(x)
                    mask = (mask == 0).float().reshape(B, 1, h, w)

                    mask_or=mask.clone()

                    masked_image = image * (mask < 0.5)
                    batch = {
                        "image": image.to(device),
                        "txt": batch_size * [txt],
                        "mask": mask.to(device),
                        "masked_image": masked_image.to(device),
                    }



                    c = model.cond_stage_model.encode(batch["txt"])
                    c_cat = list()
                    for ck in model.concat_keys:
                        cc = batch[ck].float()
                        if ck != model.masked_image_key:
                            bchw = [batch_size, 4, h // 8, w // 8]
                            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                        else:
                            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                        c_cat.append(cc)
                    c_cat = torch.cat(c_cat, dim=1)

                    # cond
                    cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                    # uncond cond
                    uc_cross = model.get_unconditional_conditioning(batch_size, "")
                    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                    shape = [model.channels, h // 8, w // 8]

                    # start code
                    _t = args.t  # 0-999
                    z = model.get_first_stage_encoding(model.encode_first_stage(image.to(device)))
                    t = torch.tensor([_t] * batch_size, device=device)
                    z_t = model.q_sample(x_start=z, t=t)

                    # target_image_b_name='/data/Adv-Diffusion-main/app_target/14.png'
                    # target_image_b = Image.open(target_image_b_name).convert(
                    #     'RGB')  # Image.open(tgt_img_path).convert('RGB')
                    # target_image_b = transform(target_image_b).unsqueeze(0).to(device)

                    #ours
                    # samples_cfg, intermediates = sampler.sample(
                    #     ddim_steps,
                    #     batch_size,
                    #     shape,
                    #     cond,
                    #     verbose=False,
                    #     eta=1.0,
                    #     unconditional_guidance_scale=scale,
                    #     unconditional_conditioning=uc_full,
                    #     x_T=z_t,
                    #     _t=_t + 1,
                    #     log_every_t=1,
                    #     classifier=classifier,
                    #     classifier_scale=classifier_scale,
                    #     x_target=tgt_image,
                    #     # x_target_b=target_image_b,
                    #     mask=mask.to(device),
                    #     x0=image.clone().to(device),
                    #     tgt_image=tgt_image,  # 传递到采样器的目标图像参数
                    #     source_img=image.clone().to(device),
                    #
                    #     # multi_scale_weight=0.3,  # 多尺度损失权重（可根据实验调整）
                    # )
                    #adv-diffusion原本的
                    samples_cfg, intermediates = sampler.sample(
                        ddim_steps,
                        batch_size,
                        shape,
                        cond,
                        verbose=False,
                        eta=1.0,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc_full,
                        x_T=z_t,
                        _t=_t + 1,
                        log_every_t=1,
                        classifier=classifier,
                        classifier_scale=classifier_scale,
                        x_target=tgt_image
                    )


                    x_samples_ddim = model.decode_first_stage(samples_cfg)
                    result = torch.clamp(x_samples_ddim, min=-1, max=1)
                    # result = image#torch.clamp(x_samples_ddim, min=-1, max=1)
                    # result_save = masked_image.to(device) + (1.- mask_or.to(device)) * result.to(device)
                    result_save = masked_image.to(device) + result.to(device) * (mask.to(device) >= 0.5)
                    # 准备目标特征
                    # tgt_features = attack_model(resize(tgt_image)).detach()
                    # 调用特征驱动的PGD攻击（替代原MI - FGSM）
                    # result = feature_driven_pgd_attack(
                    #     diffusion_model=model,
                    #     images=result_save,
                    #     original_images=image,
                    #     target_images=tgt_image,
                    #     attack_model=attack_model,
                    #     epsilon=2 / 255,  # 视觉隐蔽性内的最大扰动（可根据效果调整）
                    #     alpha=2 / 255,  # 步长=epsilon/4（经验值）
                    #     iterations=2,  # 迭代次数
                    #     device=device
                    # )

                    # 验证结果：提取特征并计算相似度
                    def calculate_similarity(img, ref_img, model):
                        """计算图像与参考图像的特征余弦相似度"""
                        img_resized = resize(img).to(device)
                        ref_img_resized = resize(ref_img).to(device)
                        with torch.no_grad():
                            feat = normalize_feature(model(img_resized))
                            ref_feat = normalize_feature(model(ref_img_resized))
                        return F.cosine_similarity(feat, ref_feat, dim=1).mean().item()

                    # 打印相似度结果（调试用）
                    # sim_with_target = calculate_similarity(result, tgt_image, attack_model)
                    # sim_with_origin = calculate_similarity(result, image, attack_model)
                    # print(f"\n对抗图像与目标图像的余弦相似度：{sim_with_target:.4f}")
                    # print(f"对抗图像与原图像的余弦相似度：{sim_with_origin:.4f}")

                  
                    # print('x_inter'+str(len(intermediates['x_inter'])))
                    # os.makedirs(os.path.join('inter',str(save_num)), exist_ok=True)
                    # for i, x_inter in enumerate(intermediates['x_inter']):
                    #     x_inter = torch.clamp(model.decode_first_stage(x_inter), min=-1, max=1)
                    #     for y, x in enumerate(range(x_inter.shape[0])):
                    #        save_image((x_inter[x] + 1) / 2, os.path.join('inter',str(save_num),f'{i}_{y}.png'))



                    os.makedirs(os.path.join(args.save+str(seed), 'img'), exist_ok=True)
                    os.makedirs(os.path.join(args.save+str(seed), 'msk'), exist_ok=True)
                    os.makedirs(os.path.join(args.save+str(seed), 'fea'), exist_ok=True)
                    print(i, batch_size)
                    for x in range(result.shape[0]):
                        print(os.path.join(args.save+str(seed), 'msk', f'{i * batch_size + x}_m.png'))
                        save_image((result[x] + 1) / 2, os.path.join(args.save+str(seed), 'img', f'{i * batch_size + x}.png'))
                        save_image((masked_image[x] + 1) / 2, os.path.join(args.save+str(seed), 'msk', f'{i * batch_size + x}_m.png'))

                    # save_image((x_inter + 1) / 2, f'res/{i}_inter.png')

                    # attack_model = attack_model_dict[attack_model_name]
                    feature1 = attack_model(resize(result)).reshape(B, -1)
                    feature2 = attack_model(resize(tgt_image)).reshape(B, -1)

                    featureO = attack_model(resize(image.to(device))).reshape(B, -1)

                    # featureadv = attack_model(resize(result_adv)).reshape(B, -1)

                    score = F.cosine_similarity(feature1, feature2)
                    print(score)
                    cos_sim_scores_dict[attack_model_name] += score.tolist()



                    score1 = F.cosine_similarity(feature1, featureO)
                    print(score1)
                    cos_sim_scores_dict_f2[attack_model_name] += score1.tolist()

                    score2 = F.cosine_similarity(feature2, featureO)
                    print(score2)

                    # scoreadv = F.cosine_similarity(featureadv, featureO)
                    # print(scoreadv)

                    def pseudocolor(feat):
                        """"""
                        r = torch.clamp(2 * feat - 1, 0, 1)  # 
                        g = torch.clamp(1 - 2 * torch.abs(feat - 0.5), 0, 1)  # 
                        b = torch.clamp(1 - 2 * feat, 0, 1)  
                        return torch.cat([r, g, b], dim=1)  
                    for x in range(B):  
                        fea1 = feature1[x]  
                        feat_dim = fea1.numel()  
                        H1 = int(torch.sqrt(torch.tensor(feat_dim)).item())
                        for possible_H in range(H1, 0, -1):
                            if feat_dim % possible_H == 0:
                                H1 = possible_H
                                W1 = feat_dim // possible_H
                                break
                        fea1_2d = fea1.reshape(1, 1, H1, W1) 
                        fea1_resized = F.interpolate(
                            fea1_2d,
                            size=(h, w),
                            mode='bilinear',
                            align_corners=False
                        )  
                        fea1_norm = (fea1_resized - fea1_resized.min()) / (
                                    fea1_resized.max() - fea1_resized.min() + 1e-8)

                        fea1_color = pseudocolor(fea1_norm) 
                        fea1_path = os.path.join(args.save + str(seed), 'fea', f'{i * batch_size + x}_feature1.png')
                        save_image(fea1_color.squeeze(0), fea1_path) 

                        feaO = featureO[x]
                        feaO_2d = feaO.reshape(1, 1, H1, W1) 
                        feaO_resized = F.interpolate(feaO_2d, size=(h, w), mode='bilinear',
                                                     align_corners=False)
                        feaO_norm = (feaO_resized - feaO_resized.min()) / (
                                    feaO_resized.max() - feaO_resized.min() + 1e-8)
                        feaO_color = pseudocolor(feaO_norm)
                        feaO_path = os.path.join(args.save + str(seed), 'fea', f'{i * batch_size + x}_featureO.png')
                        save_image(feaO_color.squeeze(0), feaO_path)

        asr_calculation(cos_sim_scores_dict,cos_sim_scores_dict_f2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='IR152')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--num', type=int, default='1000')
    parser.add_argument('--t', type=int, default=999)
    parser.add_argument('--save', type=str, default='res')
    parser.add_argument('--s', type=int, default=300)

    parser.add_argument('--model-backbone', type=str, default='resnet101',
                        choices=['resnet101', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str, default='assets/pytorch_resnet101.pth')
    parser.add_argument('--model-refine-mode', type=str, default='sampling',
                        choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)
    parser.add_argument('--model-refine-kernel-size', type=int, default=3)
    args = parser.parse_args()

    main(args)
