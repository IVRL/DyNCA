import torch
import numpy as np

from utils.loss.vector_field_loss import VectorFieldMotionLoss
from utils.loss.appearance_loss import AppearanceLoss
from utils.loss.video_motion_loss import VideoMotionLoss


class Loss(torch.nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args

        self.appearance_loss_weight = getattr(args, "appearance_loss_weight", 0.0)

        self.vector_field_motion_loss_weight = getattr(args, "vector_field_motion_loss_weight", 0.0)
        self.video_motion_loss_weight = getattr(args, "video_motion_loss_weight", 0.0)

        self.overflow_loss_weight = getattr(args, "overflow_loss_weight", 0.0)

        self._create_losses()

        self.weight_dict = self.get_manual_weight()

    def get_overflow_loss(self, input_dict, return_summary=True):
        nca_state = input_dict['nca_state']
        overflow_loss = (nca_state - nca_state.clamp(-1.0, 1.0)).abs().mean()
        return overflow_loss, None, None

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}

        if self.overflow_loss_weight != 0:
            self.loss_mapper["overflow"] = self.get_overflow_loss
            self.loss_weights["overflow"] = self.overflow_loss_weight

        if self.vector_field_motion_loss_weight != 0:
            self.loss_mapper["vector_field_motion"] = VectorFieldMotionLoss(self.args)
            self.loss_weights["vector_field_motion"] = self.vector_field_motion_loss_weight

        if self.appearance_loss_weight != 0:
            self.loss_mapper["appearance"] = AppearanceLoss(self.args)
            self.loss_weights["appearance"] = self.appearance_loss_weight

        if self.video_motion_loss_weight != 0:
            self.loss_mapper["video_motion"] = VideoMotionLoss(self.args)
            self.loss_weights["video_motion"] = self.video_motion_loss_weight

    def set_loss_weight(self, appearance_loss_log=None, loss_name='video_motion', loss_num=10.0, medium_mt=None):
        if loss_name == 'video_motion':
            img_size = self.args.img_size[0]
            img_name = self.args.target_dynamics_path.split('/')[-1].split('.')[0]
            nca_config = f"{self.args.nca_c_in}-{self.args.nca_fc_dim}"

            motion_loss_weight_reset = loss_num
            if medium_mt is not None:
                if img_size == 256:
                    motion_loss_weight_reset = min(10.0, max(medium_mt * 6.04 - 2.17, 2.0))
                elif img_size == 128:
                    motion_loss_weight_reset = min(10.0, max(medium_mt * 5.82 - 1.05, 2.0))
                if img_name in self.weight_dict[nca_config]:
                    motion_loss_weight_reset = self.weight_dict[nca_config][img_name]

            self.loss_weights["video_motion"] = motion_loss_weight_reset

        if loss_name == 'vector_field_motion':
            self.loss_weights["vector_field_motion"] = np.median(appearance_loss_log) / 50.0

    def forward(self, input_dict, return_log=True, return_summary=True):
        loss = 0
        loss_log_dict = {}
        summary_dict = {}
        for loss_name in self.loss_mapper:
            l, loss_log, sub_summary = self.loss_mapper[loss_name](input_dict, return_summary=return_summary)

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

    @staticmethod
    def get_manual_weight():
        return {'12-96': {'ants': 0.2, 'fur': 1.0, 'sea_2': 4.0, 'flames': 3.0,
                          'sky_clouds_1': 0.25,
                          'smoke_2': 0.1, 'smoke_3': 0.5, 'sea_3': 2.0, 'calm_water_4': 1.0,
                          'calm_water_2': 1.0},
                '16-128': {'ants': 0.2, 'fur': 1.0, 'sea_2': 4.0, 'flames': 2.0,
                           'sky_clouds_1': 0.25,
                           'smoke_2': 0.1, 'smoke_3': 1.0, 'sea_3': 2.0, 'calm_water_4': 1.0,
                           'calm_water_2': 1.0}}
