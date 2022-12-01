import torch
import numpy as np
from utils.loss.motion_loss import MotionLoss
from utils.loss.texture_loss import TextureLoss
from utils.loss.motion_texture_loss import MotionTextureLoss

class Loss(torch.nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args

        self.clip_loss_weight = getattr(args, "clip_loss_weight", 0.0)
        self.motion_loss_weight = getattr(args, "motion_loss_weight", 0.0)
        self.texture_loss_weight = getattr(args, "texture_loss_weight", 0.0)

        self.motion_texture_loss_weight = getattr(args, "motion_texture_loss_weight", 0.0)
        self.two_stream_loss_weight = getattr(args, "two_stream_loss_weight", 0.0)

        self.regularization_loss_weight = getattr(args, "regularization_loss_weight", 0.0)
        self.overflow_loss_weight = getattr(args, "overflow_loss_weight", 0.0)
        self.motion_texture_loss_weight_change = getattr(args, "motion_texture_loss_weight_change", 0.0)

        self._create_losses()
        
        self.img_size = args.img_size[0]
        self.img_name = args.style_path.split('/')[-1].split('.')[0]
        self.nca_config = f"{args.nca_c_in}-{args.nca_fc_dim}"

        self.weight_dict = self.get_manual_weight()

    @staticmethod
    def get_regularization_loss(input_dict, return_summary = True):
        nca_feature_list = input_dict['nca_feature_list']
        z = nca_feature_list[-1]
        return (z ** 2).mean(), None, None

    def get_overflow_loss(self, input_dict, return_summary = True):
        nca_state = input_dict['nca_state']
        overflow_loss = (nca_state - nca_state.clamp(-1.0, 1.0)).abs().mean()
        return overflow_loss, None, None

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

        if self.regularization_loss_weight != 0:
            self.loss_mapper["regularization"] = self.get_regularization_loss
            self.loss_weights["regularization"] = self.regularization_loss_weight

        if self.overflow_loss_weight != 0:
            self.loss_mapper["overflow"] = self.get_overflow_loss
            self.loss_weights["overflow"] = self.overflow_loss_weight

        if self.motion_loss_weight != 0:
            self.loss_mapper["motion"] = MotionLoss(self.args)
            self.loss_weights["motion"] = self.motion_loss_weight

        if self.texture_loss_weight != 0:
            self.loss_mapper["texture"] = TextureLoss(self.args)
            self.loss_weights["texture"] = self.texture_loss_weight
        
        if self.motion_texture_loss_weight != 0:
            self.loss_mapper["motion_texture"] = MotionTextureLoss(self.args)
            self.loss_weights["motion_texture"] = self.motion_texture_loss_weight

    def set_loss_weight(self, loss_log_before_motion=None, loss_name = 'motion_texture', loss_num = 10.0, medium_mt = None):
        if(loss_name == 'motion_texture'):
            motion_loss_weight_reset = loss_num
            if(medium_mt is not None):
                if(self.img_size == 256):
                    motion_loss_weight_reset = min(10.0, max(medium_mt * 6.04 - 2.17, 2.0))
                elif(self.img_size == 128):
                    motion_loss_weight_reset = min(10.0, max(medium_mt * 5.82 - 1.05, 2.0))
                if(self.img_name in self.weight_dict[self.nca_config]):
                        motion_loss_weight_reset = self.weight_dict[self.nca_config][self.img_name]

            print(f'Set motion texture loss weight to {motion_loss_weight_reset}')
            self.loss_weights["motion_texture"] = motion_loss_weight_reset

    def forward(self, input_dict, return_log=True, return_summary=True):
        loss = 0
        loss_log_dict = {}
        summary_dict = {}
        for loss_name in self.loss_mapper:
            l, loss_log, sub_summary = self.loss_mapper[loss_name](input_dict, return_summary = return_summary)

            
            if loss_log is not None:
                for sub_loss_name in loss_log:
                    loss_log_dict[f'{loss_name}-{sub_loss_name}'] = loss_log[sub_loss_name].item()
            
            if sub_summary is not None:
                for summary_name in sub_summary:
                    summary_dict[f'{loss_name}-{summary_name}'] = sub_summary[summary_name]

            l *= self.loss_weights[loss_name]
            loss_log_dict[loss_name] = l.item()
            loss += l

        output = [loss]
        if return_log:
            output.append(loss_log_dict)
        if return_summary:
            output.append(summary_dict)
        else:
            output.append(None)
        if len(output) == 1:
            return output[0]
        else:
            return output
    def get_manual_weight(self):
        return {'12-96':{'ants':0.2,'fur':1.0, 'sea_2':4.0, 'flames':3.0, 
                                'sky_clouds_1':0.25,
                               'smoke_2':0.1, 'smoke_3':0.5, 'sea_3':2.0, 'calm_water_4':1.0,
                                'calm_water_2':1.0},
                      '16-128':{'ants':0.2,'fur':1.0, 'sea_2':4.0, 'flames':2.0, 
                                'sky_clouds_1':0.25,
                               'smoke_2':0.1, 'smoke_3':1.0, 'sea_3':2.0, 'calm_water_4':1.0, 
                                'calm_water_2':1.0}}