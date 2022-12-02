import torch
import numpy as np
import os
import sys
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import copy

import models
from utils.misc.flow_viz import flow_to_image, plot_vec_field

class VideoMotionLoss(torch.nn.Module):
    def __init__(self, args):
        super(VideoMotionLoss, self).__init__()
        self.args = args

        args.video_motion_slw_weight = 0.0
        args.video_motion_ot_weight = 0.0
        args.video_motion_gram_weight = 0.0
        if(args.video_motion_loss_type == 'MotionOT'):
            args.video_motion_ot_weight = 1.0
        elif(args.video_motion_loss_type == 'MotionSlW'):
            args.video_motion_slw_weight = 1.0
        elif(args.video_motion_loss_type == 'MotionGram'):
            args.video_motion_gram_weight = 1.0

        self.slw_weight = args.video_motion_slw_weight
        self.ot_weight = args.video_motion_ot_weight
        self.gram_weight = args.video_motion_gram_weight
        
        self.img_size_for_loss = args.motion_img_size
        print('Image Size For VideoMotionLoss: ', self.img_size_for_loss)

        self.motion_model_name = args.motion_model_name
        self.motion_model = models.get_model(self.motion_model_name, models_path="pretrained_models/").to(
            args.DEVICE).eval()
        print(f"Successfully Loaded {self.motion_model_name} model")

        self.apply_loss = True
        
        self.temp_avg = False
        self.args.gram_avg = False

        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

        if self.slw_weight != 0:
            self.loss_mapper["motion_SlW"] = MotionSlicedWassersteinLoss(self.args)
            self.loss_weights["motion_SlW"] = self.slw_weight

        if self.ot_weight != 0:
            self.loss_mapper["motion_OT"] = MotionOptimalTransportLoss(self.args)
            self.loss_weights["motion_OT"] = self.ot_weight

        if self.gram_weight != 0:
            self.loss_mapper["motion_Gram"] = MotionGramLoss(self.args)
            self.loss_weights["motion_Gram"] = self.gram_weight
            self.loss_weights["motion_OT"] = 0.0

    def get_motion_feature_two_frames(self, image1, image2, size=(128, 128)):
        image1_size = image1.shape[2]
        image2_size = image2.shape[2]
        if (image1_size != size[0]):
            image1 = TF.resize(image1, size)
        if (image2_size != size[0]):
            image2 = TF.resize(image2, size)
        '''motion_feature is a list containing feature maps from different layers in motion model
           flow is the predicted flow that should be the same size as the input image and 2 channels and same batch size 
        '''
        # MSOEnet accepts grayscale [0,1]
        x1 = (image1 + 1.0) / 2.0
        x2 = (image2 + 1.0) / 2.0
        x1 = TF.rgb_to_grayscale(x1)
        x2 = TF.rgb_to_grayscale(x2)
        image_cat = torch.stack([x1, x2], dim=-1)
        flow, motion_feature = self.motion_model(image_cat, return_features=True)

        return motion_feature, flow


    def forward(self, input_dict, return_summary = True):
        generated_image_list = input_dict['generated_image_list_motion']
        assert len(generated_image_list) >= 2
        loss_log_dict = None

        loss = 0.0
        target_image_list = input_dict['target_motion_image_list']
        for idx in range(len(generated_image_list) - 1):
            generated_image_before_nca = generated_image_list[idx]
            generated_image_after_nca = generated_image_list[idx + 1]

            target_image_1 = target_image_list[idx]
            target_image_2 = target_image_list[idx + 1]

            motion_feature_gen_list, flow_gen = \
                                    self.get_motion_feature_two_frames(generated_image_before_nca, \
                                                                generated_image_after_nca, \
                                                                size=self.img_size_for_loss)
            with torch.no_grad():
                motion_feature_target_list, flow_target = \
                                            self.get_motion_feature_two_frames(target_image_1, \
                                                                            target_image_2, \
                                                                            size=self.img_size_for_loss)
            for loss_name in self.loss_mapper:
                loss_weight = self.loss_weights[loss_name]
                loss_func = self.loss_mapper[loss_name]
                loss += loss_weight * loss_func(motion_feature_target_list, motion_feature_gen_list)

            '''Only pick the last generated optic flow to visualize. For convenience and simple code. '''
            if(return_summary):
                flow_gen_list = [flow_to_image(flow_gen[b].permute(1, 2, 0).detach().cpu().numpy()).transpose(2, 0, 1) for b in range(len(flow_gen))]
                flow_target_list = [flow_to_image(flow_target[b].permute(1, 2, 0).detach().cpu().numpy()).transpose(2, 0, 1) for b in range(len(flow_target))]
                flow_gen_numpy = np.stack(flow_gen_list)
                flow_target_numpy = np.stack(flow_target_list)

                vec_gen_list = [np.array(plot_vec_field(flow_gen[b].detach().cpu().numpy(), name=f'Generated{torch.mean(torch.norm(flow_gen[b], dim=0)).item()}')).transpose(2, 0, 1) for b in range(len(flow_gen))]
                vec_target_list = [np.array(plot_vec_field(flow_target[b].detach().cpu().numpy(), name=f'Target{torch.mean(torch.norm(flow_target[b], dim=0)).item()}')).transpose(2, 0, 1) for b in range(len(flow_target))]
                vec_gen_numpy = np.stack(vec_gen_list)
                vec_target_numpy = np.stack(vec_target_list)

                summary = {}
                summary['target_video_flow'] = flow_target_numpy
                summary['generated_video_flow'] = flow_gen_numpy
                summary['target_video_vec'] = vec_target_numpy
                summary['generated_video_vec'] = vec_gen_numpy
                return loss, loss_log_dict, summary
            else:
                return loss, loss_log_dict, None
                
                    
class MotionSlicedWassersteinLoss(torch.nn.Module):
    def __init__(self, args):
        super(MotionSlicedWassersteinLoss, self).__init__()
        self.args = args

    @staticmethod
    def project_sort(x, proj):
        return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

    def sliced_ot_loss(self, source, target, proj_n=32):
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n), dim=0).to(self.args.DEVICE)
        source_proj = MotionSlicedWassersteinLoss.project_sort(source, projs)
        target_proj = MotionSlicedWassersteinLoss.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, n, mode='nearest')
        return (source_proj - target_interp).square().sum()

    def forward(self, target_features, generated_features):
        loss = 0.0
        for x, y in zip(generated_features, target_features):
            b,c,h,w = x.shape
            x = x.reshape(b,c,h*w)
            y = y.reshape(b,c,h*w)
            loss += self.sliced_ot_loss(x, y)
        return loss
        
        
class MotionOptimalTransportLoss(torch.nn.Module):
    def __init__(self, args):
        super(MotionOptimalTransportLoss, self).__init__()
        self.args = args

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
        dist = 1. - torch.mm(x, y_t) / (x_norm + 1e-10) / (y_norm + 1e-10)
        return dist

    @staticmethod
    def style_loss_cos(X, Y, cos_d=True):
        # X,Y: 1*d*N*1
        d = X.shape[1]

        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)  # N*d
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

        # Relaxed EMD
        CX_M = MotionOptimalTransportLoss.pairwise_distances_cos(X, Y)

        m1, m1_inds = CX_M.min(1)
        m2, m2_inds = CX_M.min(0)

        remd = torch.max(m1.mean(), m2.mean())

        return remd

    @staticmethod
    def moment_loss(X, Y):  # matching mean and cov
        X = X.squeeze().t()
        Y = Y.squeeze().t()

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = mu_d + D_cov

        return loss

    @staticmethod
    def get_ot_loss_single_batch(x_feature, y_feature):
        randomize = True
        loss = 0
        for x, y in zip(x_feature, y_feature):
            c = x.shape[1]
            h, w = x.shape[2], x.shape[3]
            x = x.reshape(1, c, -1, 1)
            y = y.reshape(1, c, -1, 1)
            if h > 32 and randomize:
                indices = np.random.choice(np.arange(h * w), size=1000, replace=False)
                indices = np.sort(indices)
                indices = torch.LongTensor(indices)
                x = x[:, :, indices, :]
                y = y[:, :, indices, :]
            loss += MotionOptimalTransportLoss.style_loss_cos(x, y)
            loss += MotionOptimalTransportLoss.moment_loss(x, y)
        return loss

    def forward(self, target_features, generated_features):
        batch_size = target_features[0].shape[0]
        loss = 0.0
        for b in range(batch_size):
            target_feature = [t[b:b + 1] for t in target_features]
            generated_feature = [g[b:b + 1] for g in generated_features]
            loss += self.get_ot_loss_single_batch(target_feature, generated_feature)
        return loss / batch_size
    
class MotionGramLoss(torch.nn.Module):
    def __init__(self, args):
        super(MotionGramLoss, self).__init__()
        self.args = args

    @staticmethod
    def get_gram(y):
        b, c, h, w = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        grams = features.bmm(features_t) / (c * h * w)
        return grams

    def forward(self, target_features, generated_features):
        loss = 0.0
        for target_feature, generated_feature in zip(target_features, generated_features):
            if(self.args.gram_avg):
                gram_target = target_feature
            else:
                gram_target = self.get_gram(target_feature)
            gram_generated = self.get_gram(generated_feature)
            loss = loss + (gram_target - gram_generated).square().mean()
        return loss

def get_gram(y):
    b, c, h, w = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    grams = features.bmm(features_t) / (h * w)
    return grams
