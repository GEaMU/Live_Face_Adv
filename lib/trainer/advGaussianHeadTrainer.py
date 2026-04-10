import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import lpips
import os
import torch
import numpy as np
import os
import cv2
import trimesh
import pyrender
import time

from recogModel import face_cos

class advGaussianHeadTrainer():
    def __init__(self, dataloader, delta_poses, gaussianhead, supres, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.delta_poses = delta_poses
        self.gaussianhead = gaussianhead
        self.supres = supres
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        #self.device = torch.device('cuda:%d' % gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)



    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):
                
                # prepare data
                to_cuda = ['images', 'masks', 'visibles', 'images_coarse', 'masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                images = data['images']
                visibles = data['visibles']
                if self.supres is None:
                    images_coarse = images
                    visibles_coarse = visibles
                else:
                    images_coarse = data['images_coarse']
                    visibles_coarse = data['visibles_coarse']

                resolution_coarse = images_coarse.shape[2]
                resolution_fine = images.shape[2]

                data['pose'] = data['pose'] + self.delta_poses[data['exp_id'], :]

                # render coarse images
                data = self.gaussianhead.generate(data)
                data = self.camera.render_gaussian(data, resolution_coarse)
                render_images = data['render_images']

                #crop images for augmentation
                scale_factor = random.random() * 0.45 + 0.8
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images

                # generate super resolution images
                supres_images = self.supres(cropped_render_images)
                data['supres_images'] = supres_images

                # # crop images for augmentation
                # scale_factor = random.random() * 0.45 + 0.8
                # scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                # cropped_render_images, cropped_images, cropped_visibles = render_images, images,visibles
                # data['cropped_images'] = cropped_images
                #
                # # generate super resolution images
                # supres_images = self.supres(cropped_render_images)
                # data['supres_images'] = supres_images

                #-------------------------------------------------------
                image_path=data['image_path']
                try:
                    image_feture = face_cos.get_feture(image_path[0])
                except:
                    image_feture = face_cos.get_feture("/data/0two/Gaussian-Head-Avatar-main/mini_demo_dataset/017/images/0000/image_222200037.jpg")

                #target_path="/data/0two/Gaussian-Head-Avatar-main/mini_demo_dataset/031/images/0300/image_222200037.jpg"
                #target_feture = face_cos.get_feture(target_path)

                image = data['images'][0].permute(1, 2, 0).detach().cpu().numpy()
                image = (image * 255).astype(np.uint8)[:, :, ::-1]
                #
                render_image = data['render_images'][0, 0:3].permute(1, 2, 0).detach().cpu().numpy()
                render_image = (render_image * 255).astype(np.uint8)[:, :, ::-1]
                #
                # cropped_image = data['cropped_images'][0].permute(1, 2, 0).detach().cpu().numpy()
                # cropped_image = (cropped_image * 255).astype(np.uint8)[:, :, ::-1]

                # supres_image = data['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
                # supres_image = (supres_image * 255).astype(np.uint8)[:, :, ::-1]

                render_image = cv2.resize(render_image, (image.shape[0], image.shape[1]))
                #result = np.hstack((image, render_image, cropped_image, supres_image))

                #name='train_advfreeview'
                result_path='results/gaussianhead_train017'
                iter=idx + epoch * len(self.dataloader)

                cv2.imwrite('%s/%06d.jpg' % (result_path, iter), render_image)
                #time.sleep(1)
                file_path = '%s/%06d.jpg' % (result_path, iter)
                start_time=time.time()
                timeout=60
                check_interval=1
                while not os.path.exists(file_path):
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"等待文件 {file_path} 超时（{timeout} 秒）")
                    print(f"文件 {file_path} 不存在，等待 {check_interval} 秒后重试...")
                    time.sleep(check_interval)
                print(f"文件 {file_path} 已存在，继续执行！")
                result_path_t='%s/%06d.jpg' % (result_path, iter)
                try:
                    general_feature=face_cos.get_feture(result_path_t)
                except:
                    general_feature=face_cos.get_feture('%s/%06d.jpg' % (result_path, 0))


                # loss_fnn1 = torch.nn.CosineEmbeddingLoss()
                # loss_flag = torch.tensor([1.0]).cuda()
                loss_fnn2 = torch.nn.CosineEmbeddingLoss()
                loss_flag1= torch.tensor([-1.0]).cuda()
                #loss_feture_tar=loss_fnn1(general_feature.unsqueeze(0),target_feture.unsqueeze(0),loss_flag)
                loss_feture_ori=loss_fnn2(image_feture.unsqueeze(0),general_feature.unsqueeze(0),loss_flag1)

                max_perturb = 0.1
                relative_perturb = data['adv_color'] / (data['color'].abs() + 1e-6)  # 相对扰动比例
                exceed_mask = (relative_perturb.norm(dim=-1) > max_perturb).float()
                #loss_magnitude = torch.sum(exceed_mask * (relative_perturb.norm(dim=-1) - max_perturb) ** 2)

#-----------------------------------------------------------------------------------------------------------
                # loss functions
                loss_rgb_lr =1 #F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                loss_rgb_hr =1# F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                            (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
                #loss = loss_rgb_hr + loss_rgb_lr + loss_vgg * 1e-1+loss_feture_ori*1+loss_magnitude#loss_feture_tar*100+
                loss = loss_vgg * 1e-1+loss_feture_ori*100#+loss_magnitude#loss_feture_tar*100+
                print(loss_vgg,loss_feture_ori,loss)#loss_feture_tar,

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = {
                    'data': data,
                    'delta_poses' : self.delta_poses,
                    'gaussianhead' : self.gaussianhead,
                    'supres' : self.supres,
                    'loss_rgb_lr' : loss_rgb_lr,
                    'loss_rgb_hr' : loss_rgb_hr,
                    'loss_vgg' : loss_vgg,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader),
                    'loss_feture':loss_feture_ori
                }
                self.recorder.log(log)


    def random_crop(self, render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine):
        render_images_scaled = F.interpolate(render_images, scale_factor=scale_factor)
        images_scaled = F.interpolate(images, scale_factor=scale_factor)
        visibles_scaled = F.interpolate(visibles, scale_factor=scale_factor)

        if scale_factor < 1:
            render_images = torch.ones([render_images_scaled.shape[0], render_images_scaled.shape[1], resolution_coarse, resolution_coarse], device=self.device)
            left_up_coarse = (random.randint(0, resolution_coarse - render_images_scaled.shape[2]), random.randint(0, resolution_coarse - render_images_scaled.shape[3]))
            render_images[:, :, left_up_coarse[0]: left_up_coarse[0] + render_images_scaled.shape[2], left_up_coarse[1]: left_up_coarse[1] + render_images_scaled.shape[3]] = render_images_scaled

            images = torch.ones([images_scaled.shape[0], images_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            visibles = torch.ones([visibles_scaled.shape[0], visibles_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images[:, :, left_up_fine[0]: left_up_fine[0] + images_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + images_scaled.shape[3]] = images_scaled
            visibles[:, :, left_up_fine[0]: left_up_fine[0] + visibles_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + visibles_scaled.shape[3]] = visibles_scaled
        else:
            left_up_coarse = (random.randint(0, render_images_scaled.shape[2] - resolution_coarse), random.randint(0, render_images_scaled.shape[3] - resolution_coarse))
            render_images = render_images_scaled[:, :, left_up_coarse[0]: left_up_coarse[0] + resolution_coarse, left_up_coarse[1]: left_up_coarse[1] + resolution_coarse]

            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images = images_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
            visibles = visibles_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
        
        return render_images, images, visibles