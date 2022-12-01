import torch
import os
import sys
import torchvision.transforms.functional as TF
import torch.nn.functional as F


base_path = os.path.abspath(__file__)
base_path = base_path[:base_path.find("utils/loss/")]  # the root folder of this project
sys.path.append(base_path)  # this is for import models, to find models folder correctly
sys.path.append(base_path + "models/RAFT/core")  # this is for loading raft model, for the raft.py to find the file in the same folder correctly
import models
from utils.misc.flow_viz import flow_to_image

# raft = models.get_raft_model('raft-things') # test whether could get raft model in this file

class MotionLoss(torch.nn.Module):
    def __init__(self, args):
        super(MotionLoss, self).__init__()
        self.update_args(args)

        self.args = args

        self.user_specified_motion_field = args.user_specified_motion_field
        self.direction_constrain = args.direction_constrain
        self.motion_strength_weight = args.motion_strength_weight
        self.direction_same_weight = args.direction_same_weight
        self.direction_weight = args.direction_weight
        self.motion_mse_loss_weight = args.motion_mse_loss_weight

        self.motion_loss_iteration = args.motion_loss_iteration
        self.direction_loss_iteration = args.direction_loss_iteration
        assert self.direction_loss_iteration <= self.motion_loss_iteration

        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.target_motion_vec = args.target_motion_vec
        self.target_motion_strength = args.target_motion_strength
        if(self.args.mse_simple_motion):
            self.target_motion_vec /= torch.norm(self.target_motion_vec)
            self.target_motion_vec *= self.target_motion_strength

        self.img_size_for_loss = args.img_size

        self.motion_model_name = args.motion_model_name
        self.motion_model = models.get_model(self.motion_model_name, models_path="pretrained_models/").to(
            args.DEVICE).eval()
        print(f"Successfully Loaded {self.motion_model_name} model")

        self.apply_loss = False
        self.apply_direction_constrain_loss = False

        self._create_losses()

    def update_args(self, args):
        if (args.motion_loss_weight > 0):
            args.motion_loss = True
            args.direction_constrain = False
            self.direction_constrain = False
            self.direction_change = False
            inverse_y = False
            if("two_stream" in args.motion_model_name):
                inverse_y = True
            if (args.user_specified_motion_field == True):
                self.direction_change = True
                self.direction_constrain = True
                print('Manually set motion vector field')
                assert args.motion_field_name is not None
                if(args.mse_simple_motion == False):
                    target_motion_vec = get_motion_field(args.motion_field_name, img_size=args.img_size, inverse_y = inverse_y)
                else:
                    target_motion_vec = get_motion_field(args.motion_field_name, img_size=args.img_size, flatten = False, inverse_y = inverse_y)
                target_motion_vec = target_motion_vec.to(args.DEVICE)
                args.target_motion_vec = target_motion_vec
            else:
                # if (args.motion_direction >= 0):
                #     args.direction_constrain = True
                #     self.direction_constrain = True
                #     print(f'Constrain the direction of movements to {args.motion_direction}')
                #     target_motion_vec = get_motion_field('angle', img_size=args.img_size,
                #                                          motion_direction=args.motion_direction)
                #     target_motion_vec = target_motion_vec.to(args.DEVICE)
                # elif (args.motion_direction < 0):
                target_motion_vec = None
                # target_motion_vec = get_motion_field('angle', img_size=args.img_size,
                #                                      motion_direction=None)
                direction_loss = torch.tensor(0.0)
                # target_motion_vec = target_motion_vec.to(args.DEVICE)
                args.target_motion_vec = target_motion_vec

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}
        
        if(self.args.mse_simple_motion == False):
            if(self.args.clip_loss_weight == 0 or self.direction_constrain == False): # if CLIP loss + direction constrain then there is no strength enhancement.
                self.loss_mapper['strength_diff'] = self.get_motion_strength_loss
                self.loss_weights['strength_diff'] = self.motion_strength_weight
                if(self.direction_constrain == False):
                    self.loss_weights['strength_diff'] = 1.0

            if (self.user_specified_motion_field):
                self.loss_mapper['direction_loss'] = self.get_cosine_dist
                self.loss_weights['direction_loss'] = self.direction_weight
            elif (self.user_specified_motion_field == False):
                if (self.direction_constrain):
                    pass
                    self.loss_mapper['direction_loss'] = self.get_cosine_dist
                    self.loss_weights['direction_loss'] = self.direction_weight
                pass
                # if(self.direction_constrain and self.direction_same_weight > 0):
                #     self.loss_mapper['direction_same_loss'] = MotionLoss.get_motion_variance
                #     self.loss_weights['direction_same_loss'] = self.direction_same_weight
        else:
            self.loss_mapper['mse_motion_loss'] = self.get_motion_mse_loss
            self.loss_weights['mse_motion_loss'] = self.motion_mse_loss_weight
    
    def get_motion_mse_loss(self, optic_flow, optic_flow_direction):
        optic_flow_dist = F.mse_loss(optic_flow, self.target_motion_vec)
        return optic_flow_dist

    @staticmethod
    def get_motion_variance(optic_flow, optic_flow_direction):
        direction_var = total_variance(optic_flow_direction)
        direction_same_loss = torch.mean(direction_var)
        return direction_same_loss

    def get_motion_strength_loss(self, optic_flow, optic_flow_direction):
        motion_strength = torch.norm(optic_flow, dim=1)
        # if(self.direction_constrain): #  or self.direction_change
        #     # motion_strength_loss = torch.abs(self.target_motion_strength - torch.mean(motion_strength))
        #     motion_strength_loss = (self.target_motion_strength - torch.mean(motion_strength)) ** 2
        # else:
        #     # motion_strength_loss = torch.abs(self.target_motion_strength - torch.mean(motion_strength))
        if('raft' in self.motion_model_name):
            motion_strength_loss = (self.target_motion_strength - torch.mean(motion_strength)) ** 2
        elif('flownet' in self.motion_model_name):
            motion_strength_loss = torch.abs(self.target_motion_strength - torch.mean(motion_strength))
        elif 'two_stream' in self.motion_model_name:
            motion_strength_loss = (self.target_motion_strength - torch.mean(motion_strength)) ** 2
        return motion_strength_loss

    def get_cosine_dist(self, optic_flow, optic_flow_direction):
        optic_flow_direction_1dim = optic_flow_direction.reshape(self.b_opt, self.c_opt, -1)
        direction_cos_sim = self.cos_sim(optic_flow_direction_1dim, self.target_motion_vec)
        direction_loss = 1.0 - torch.mean(direction_cos_sim)
        # direction_loss = F.mse_loss(optic_flow_direction, self.target_motion_vec)
        return direction_loss

    def update_losses_to_apply(self, epoch):
        if (epoch >= self.motion_loss_iteration):
            self.apply_loss = True
        else:
            self.apply_loss = False
        # if(epoch >= self.direction_loss_iteration and self.args.direction_constrain):
        #     self.apply_direction_constrain_loss = True
        # else:
        #     self.apply_direction_constrain_loss = False

    def get_opticflow(self, image1, image2, size=(128, 128)):
        image1_size = image1.shape[2]
        image2_size = image2.shape[2]
        if (image1_size != size[0]):
            image1 = TF.resize(image1, size)
        if (image2_size != size[0]):
            image2 = TF.resize(image2, size)
        if('raft' in self.motion_model_name):
            flow_low, flow_up = self.motion_model(image1, image2, iters=12, test_mode=True)
            return flow_up
        elif('flownet' in self.motion_model_name):
            x1 = (image1 + 1.0) / 2.0
            x2 = (image2 + 1.0) / 2.0
            img_mean = torch.tensor([0.411, 0.432, 0.45])[:, None, None].to(self.args.DEVICE)
            x1 = x1 - img_mean
            x2 = x2 - img_mean
            image_cat = torch.cat([x1, x2], dim=1)
            flow, _ = self.motion_model(image_cat, return_motion_features=True)
            return flow
        elif 'two_stream' in self.motion_model_name:
            # MSOEnet accepts grayscale [0,1]
            x1 = (image1 + 1.0) / 2.0
            x2 = (image2 + 1.0) / 2.0
            x1 = TF.rgb_to_grayscale(x1)
            x2 = TF.rgb_to_grayscale(x2)
            image_cat = torch.stack([x1, x2], dim=-1)
            flow, motion_feature = self.motion_model(image_cat, return_features=True)
            return flow

    def forward(self, input_dict, return_summary = True):
        if (self.apply_loss == False):
            return torch.tensor(0.0).to(self.args.DEVICE), None, None
        else:
            if('generated_image_before_nca' in input_dict):
                generated_image_before_nca = input_dict['generated_image_before_nca']
                generated_image_after_nca = input_dict['generated_image_after_nca']
                # b,c,h,w = generated_image_before_nca.shape
                # _,_,size = self.target_motion_vec.shape
                # if(h*w != size):
                #     self.target_motion_vec = get_motion_field(self.args.motion_field_name, img_size=[h, w])
                #     self.target_motion_vec = self.target_motion_vec.to(self.args.DEVICE)
                optic_flow = self.get_opticflow(generated_image_before_nca, generated_image_after_nca,
                                                size=self.img_size_for_loss)
                optic_flow_normalized_direction = optic_flow / torch.norm(optic_flow, dim=1, keepdim=True)
                self.b_opt, self.c_opt, _, _ = optic_flow.shape
                loss = 0
                loss_log_dict = {}
                for loss_name in self.loss_mapper:
                    loss_weight = self.loss_weights[loss_name]
                    loss_func = self.loss_mapper[loss_name]
                    cur_loss = loss_func(optic_flow, optic_flow_normalized_direction)
                    loss_log_dict[loss_name] = cur_loss
                    loss += loss_weight * cur_loss

                return loss, loss_log_dict, None
            elif('generated_image_list' in input_dict):
                generated_image_list = input_dict['generated_image_list']
                assert len(generated_image_list) >= 2
                loss = 0.0
                for idx in range(len(generated_image_list) - 1):
                    generated_image_before_nca = generated_image_list[idx]
                    generated_image_after_nca = generated_image_list[idx + 1]
                    # b,c,h,w = generated_image_before_nca.shape
                    # _,_,size = self.target_motion_vec.shape
                    # if(h*w != size):
                    #     self.target_motion_vec = get_motion_field(self.args.motion_field_name, img_size=[h, w])
                    #     self.target_motion_vec = self.target_motion_vec.to(self.args.DEVICE)
                    optic_flow = self.get_opticflow(generated_image_before_nca, generated_image_after_nca,
                                                    size=self.img_size_for_loss)
                    optic_flow_normalized_direction = optic_flow / torch.norm(optic_flow, dim=1, keepdim=True)
                    self.b_opt, self.c_opt, _, _ = optic_flow.shape
                    
                    loss_log_dict = {}
                    for loss_name in self.loss_mapper:
                        loss_weight = self.loss_weights[loss_name]
                        loss_func = self.loss_mapper[loss_name]
                        cur_loss = loss_func(optic_flow, optic_flow_normalized_direction)
                        loss_log_dict[loss_name] = cur_loss
                        loss += loss_weight * cur_loss
                loss = loss / (len(generated_image_list))
                return loss, loss_log_dict, None


def total_variance(images):
    # The input is a batch of images with shape: batch, 2, size, size

    # Calculate the difference of neighboring pixel-values.
    # The images are shifted one pixel along the height and width by slicing.
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]

    # Only sum for the last 3 axis.
    # This results in a 1-D tensor with the total variation for each image.
    sum_axis = [1, 2, 3]

    tot_var = (
            torch.sum(torch.abs(pixel_dif1), axis=sum_axis) +
            torch.sum(torch.abs(pixel_dif2), axis=sum_axis))
    return tot_var / (images.shape[2] * images.shape[3])


def get_motion_field(motion_field_name, img_size=[128, 128], flatten = True, inverse_y = False):
    try:
        motion_direction = int(motion_field_name)
        simple_direction = True
    except:
        simple_direction = False
    if (simple_direction):
        motion_direction = int(motion_field_name)
        torch_pi = torch.FloatTensor([3.1416])
        motion_rad = motion_direction / 180.0 * torch_pi
        target_motion_vec = torch.zeros((1, 2, img_size[0], img_size[1]))
        target_motion_vec[:, 0, ...] = torch.cos(motion_rad)
        target_motion_vec[:, 1, ...] = torch.sin(motion_rad)
        # target_motion_vec *= 20
        if(flatten):
            target_motion_vec = target_motion_vec.reshape(1, 2, -1)
        return target_motion_vec
    # else:
    #     motion_direction = 90.0
    #     torch_pi = torch.FloatTensor([3.1416])
    #     motion_rad = motion_direction / 180.0 * torch_pi
    #     target_motion_vec = torch.zeros((1, 2, img_size[0], img_size[1]))
    #     target_motion_vec[:, 0, ...] = torch.cos(motion_rad)
    #     target_motion_vec[:, 1, ...] = torch.sin(motion_rad)
    #     target_motion_vec *= 20
    #     return target_motion_vec

    target_motion_vec = torch.zeros((1, 2, img_size[0], img_size[1]))
    if (motion_field_name == 'circle'):
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if (radius == 0):
                    continue
                cosine = i / radius
                sine = j / radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = cosine  # sine + pi/2
                target_motion_vec[0, 1, center_x + i, center_y + j] = -sine  # cosine + pi/2
    elif (motion_field_name == 'concentrate'):
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if (radius == 0):
                    continue
                cosine = i / radius
                sine = j / radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = -sine # 
                target_motion_vec[0, 1, center_x + i, center_y + j] = -cosine #
    elif (motion_field_name == 'diverge'):
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                radius = (i ** 2 + j ** 2) ** 0.5
                if (radius == 0):
                    continue
                cosine = i / radius
                sine = j / radius
                target_motion_vec[0, 0, center_x + i, center_y + j] = sine # 
                target_motion_vec[0, 1, center_x + i, center_y + j] = cosine #
    elif (motion_field_name == '2block'):
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if (i >= 0 and j >= 0):
                    rad = 0
                elif (i < 0 and j < 0):
                    rad = 180
                elif (i >= 0 and j < 0):
                    rad = 0
                elif (i < 0 and j >= 0):
                    rad = 180
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)
    elif (motion_field_name == '3block'):
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if (i >= 0 and j >= 0):
                    rad = 0
                elif (i < 0 and j < 0):
                    rad = 90
                elif (i >= 0 and j < 0):
                    rad = 0
                elif (i < 0 and j >= 0):
                    rad = 180
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)
    elif (motion_field_name == '4block'):
        center_x = img_size[0] // 2
        center_y = img_size[1] // 2
        torch_pi = torch.FloatTensor([3.1416])

        for i in range(-center_x, center_x):
            for j in range(-center_y, center_y):
                if (i >= 0 and j >= 0):
                    rad = 0
                elif (i < 0 and j < 0):
                    rad = 180
                elif (i >= 0 and j < 0):
                    rad = 90
                elif (i < 0 and j >= 0):
                    rad = 270
                motion_rad = rad / 180.0 * torch_pi
                target_motion_vec[0, 0, center_x + i, center_y + j] = torch.cos(motion_rad)
                target_motion_vec[0, 1, center_x + i, center_y + j] = torch.sin(motion_rad)
    else:
        print('Not Implemented Motion Field')
        exit()
    if(flatten):
        target_motion_vec = target_motion_vec.reshape(1, 2, -1)
    if(inverse_y):
        target_motion_vec[:, 0, ...] = -target_motion_vec[:, 0, ...]
    return target_motion_vec


if (__name__ == '__main__'):
#     class Args():
#         def __init__(self, config):
#             for k in config.keys():
#                 setattr(self, k, config[k])


#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     args_dict = {
#         'user_specified_motion_field': False,
#         'direction_constrain': True,
#         'motion_strength_weight': 1.0,
#         'direction_same_weight': 1.0,
#         'direction_weight': 10.0,
#         'motion_loss_iteration': 10,
#         'target_motion_vec': torch.randn(1, 2, 1).to(DEVICE),
#         'img_size': [128, 128],
#         'motion_model_name': 'raft-things',
#         'device': DEVICE
#     }
#     args_class = Args(args_dict)
#     motion_loss_class = MotionLoss(args_class)
#     motion_loss_class.update_losses_to_apply(20)
#     loss = motion_loss_class(torch.randn(3, 3, 256, 256).to(DEVICE), torch.randn(3, 3, 256, 256).to(DEVICE))
    mf = get_motion_field("concentrate", img_size=[128, 128])
    mf = mf / torch.norm(mf)
    mf *= 5.0
    print(torch.norm(mf))
    
