import os

import warnings

import json
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy

from collections import defaultdict

from models.dynca import DyNCA

from utils.misc.display_utils import plot_train_log, save_train_image
from utils.misc.preprocess_texture import preprocess_style_image

from utils.loss.loss import Loss

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
torch.backends.cudnn.deterministic = True

import argparse

parser = argparse.ArgumentParser(
    description='Dynamic Image generation using Texture Loss or Activation Maximization with DeepNCA/NCA')

# Add the arguments
parser.add_argument("--motion_img_size", nargs=2, type=int,
                    help="Image size (width height) to compute motion loss | default = (256, 256)",
                    default=[128, 128], dest='motion_img_size')
parser.add_argument("--texture_img_size", nargs=2, type=int,
                    help="Image size (width height) to compute texture loss| default = (256, 256)",
                    default=[128, 128], dest='texture_img_size')
parser.add_argument("--nca_seed_size", nargs=2, type=int,
                    help="Seed size of NCA model (width, height) | default = (256, 256)",
                    default=[128, 128], dest='nca_seed_size')
parser.add_argument("--output_dir", type=str, help="Output directory", default="out/VectorFieldMotion/",
                    dest='output_dir')
parser.add_argument("--video_length", type=float, help="Video length in seconds (not interpolated)", default=10,
                    dest='video_length')
parser.add_argument("--video_only", action='store_true', help="Only generate video using pretrained model",
                    dest='video_only')

# Target
parser.add_argument("--exp_type", type=str,
                    help="The type of experiment, texture generation (texture) or activation maximization (act)",
                    default='texture',
                    dest='exp_type')
parser.add_argument("--style_path", type=str, help="Path to style image", default='./image/style/bubble.jpg',
                    dest='style_path')
parser.add_argument("--content_path", type=str, help="Path to content image", default='',
                    dest='content_path')
parser.add_argument("--layer_act", type=str,
                    help="The channel to maximize, e.g., mixed4a-8 is the 8th channel at layer mixed4a",
                    default='mixed4a-1', dest='layer_act')
parser.add_argument("--resize_latent",
                    help="Whether to resize the nca result to the wanted latent code size. Only valid with enlarged latent code fitting",
                    default=False, action='store_true', dest='resize_latent')

# Models
parser.add_argument("--vqae_model", type=str, default='ldm-vq-f8',
                    help="VQAE model name. If not specified a normal NCA will be trained",
                    dest='vqae_model')
parser.add_argument("--vqae_quantization", action='store_true', help="Perform quantization during the optimization",
                    dest='vqae_quantization')

# NCA
parser.add_argument("--nca_num_trainable_conv", type=int, help="Number of trainable conv channels in the NCA",
                    default=0,
                    dest='nca_num_trainable_conv')
parser.add_argument("--condition_nca_channel", type=int, help="The condition channel in NCA, 0 means no condition",
                    default=0,
                    dest='condition_nca_channel')
parser.add_argument("--nca_pool_size", type=int, help="Number of elements in the NCA pool", default=128,
                    dest='nca_pool_size')
parser.add_argument("--nca_step_range", nargs=2, type=int, help="Range of steps to apply NCA (32, 96)",
                    default=[32, 96], dest='nca_step_range')
parser.add_argument("--nca_inject_seed_step", type=int, help="Inject seed every time after this many iterations",
                    default=8, dest='nca_inject_seed_step')
parser.add_argument("--nca_channels", type=int, help="Number of Channels in the NCA model", default=16, dest='nca_c_in')
parser.add_argument("--nca_fc_dim", type=int, help="FC layer dimension", default=64, dest='nca_fc_dim')
parser.add_argument("--nca_filter_scale", type=float, help="Scaling factor of the NCA filters", default=1.0,
                    dest='nca_filter_scale')
parser.add_argument("--nca_seed_mode", type=str, help="Scaling factor of the NCA filters", default='random',
                    choices=DyNCA.SEED_MODES, dest='nca_seed_mode')
parser.add_argument("--nca_random_seed", type=int, help="Random seed used to initialize NCA pool", default=None,
                    dest='nca_random_seed')
parser.add_argument("--nca_hard_clamping", action='store_true', help="Hard Clamping of NCA output to [-1.0, 1.0] range",
                    dest='nca_hard_clamping')
parser.add_argument("--nca_padding_mode", type=str, default='constant',
                    help="Padding mode when NCA cells are perceiving",
                    choices=['constant', 'reflect', 'replicate', 'circular'],
                    dest='nca_padding_mode')
parser.add_argument("--nca_delta_T_std", type=float, help="Standard deviation of delta_T", default=0.3,
                    dest='nca_delta_T_std')
parser.add_argument("--nca_half_channel", type=int,
                    help="1 means only first 3 channel will be used to produce the RGB output", default=0,
                    dest='nca_half_channel')
parser.add_argument("--nca_pos_emb", type=str, default='CPE', choices=['None', 'SPE', 'CPE', 'PPE'],
                    help="The positional embedding mode to use. CPE (Cartesian), SPE (Sinusoidal), PPE (Pole-based), or None",
                    dest='nca_pos_emb')
parser.add_argument("--ppe_num_poles", type=int, default=5,
                    help="Number of poles in the pole-based positional encoding",
                    dest='ppe_num_poles')
parser.add_argument("--pe_normalize", type=int,
                    help="1 means normalize the positional embedding", default=1,
                    dest='pe_normalize')

parser.add_argument("--nca_linear_perturb", action='store_true',
                    help="Whether to linear perturbation to all cell states (Adv Programming)",
                    dest='nca_linear_perturb')
parser.add_argument("--scales", nargs='+', action='append', type=int,
                    help="Specify the scales of layered NCA. The last scale must be 1 indicating the original size.",
                    default=[], dest='scales')
parser.add_argument("--nca_perception_scales", nargs='+', action='append', type=int,
                    help="Specify the scales at which the NCA perception will be performed.",
                    default=[], dest='nca_perception_scales')

# Loss Function
# Texture
parser.add_argument("--texture_loss_weight", type=float,
                    help="Coefficient of Loss used for Texture or Activation Maximization", default=0.0,
                    dest='texture_loss_weight')
parser.add_argument("--texture_loss_type", type=str,
                    help="The method to compute style loss. Sliced W-distance Loss, OT (Optimal transport), Gram",
                    choices=["SlW", "OT", "Gram"],
                    default="OT",
                    dest='texture_loss_type')
parser.add_argument("--texture_model", type=str,
                    help="The model to compute style loss. vgg, vqae, Gaussian pyramid", default="vgg",
                    dest='texture_model')
# Motion
parser.add_argument("--motion_loss_weight", type=float, help="Coefficient of Motion Loss", default=0.025,
                    dest='motion_loss_weight')
parser.add_argument("--motion_strength_weight", type=float, help="Coefficient of Motion enhancing loss", default=1.0,
                    dest='motion_strength_weight')

parser.add_argument("--direction_weight", type=float, help="Coefficient of direction indicating loss", default=10.0,
                    dest='direction_weight')
parser.add_argument("--motion_mse_loss_weight", type=float,
                    help="Coefficient of MSE loss between generated optic flow and target motion field", default=1.0,
                    dest='motion_mse_loss_weight')
parser.add_argument("--motion_direction", type=float, help="Direction of motion, 0-360 degrees", default=-1,
                    dest='motion_direction')
parser.add_argument("--motion_field_name", type=str,
                    help="Name of the motion vector field to be used", default=None,
                    dest='motion_field_name')
parser.add_argument("--motion_model_name", type=str, default='raft-things',
                    help="Optic Flow computing model. Default is RAFT things. Note that for RAFT model the input should be in range [-1,1]",
                    dest='motion_model_name')
parser.add_argument("--motion_loss_iteration", type=int,
                    help="Number of iterations after which the motion loss is used",
                    default=2000,
                    dest='motion_loss_iteration')
parser.add_argument("--direction_loss_iteration", type=int,
                    help="Number of iterations after which the direction constrain loss is used",
                    default=1000,
                    dest='direction_loss_iteration')
parser.add_argument("--motion_texture_loss_weight", type=float,
                    help="Coefficient of Motion Texture Loss for temporal texture synthesis. Not needed in this training code",
                    default=0,
                    dest='motion_texture_loss_weight')
parser.add_argument("--target_motion_strength", type=float,
                    help="The value of target norm of the optic flow/motion field", default=5.0,
                    dest='target_motion_strength')
parser.add_argument("--motion_weight_change_interval", type=int,
                    help="Interval of iterations for changing the motion loss weight. ",
                    default=500,
                    dest='motion_weight_change_interval')
parser.add_argument("--nca_base_num_steps", type=float,
                    help="Number of NCA steps to normalize the magnitude of the optic flow.",
                    default=24.0,
                    dest='nca_base_num_steps')
# CLIP
parser.add_argument("--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32',
                    dest='clip_model')
parser.add_argument("--prompt", type=str, help="Prompt for CLIP guided image generation",
                    default="meteor shower",
                    dest='prompt')
parser.add_argument("--clip_loss_weight", type=float, help="Coefficient of CLIP guided image generation loss",
                    default=0.0,
                    dest='clip_loss_weight')
parser.add_argument("--clip_augments", nargs='+', action='append', type=str,
                    choices=['Ji', 'Sh', 'Gn', 'Pe', 'Ro', 'Af', 'Et', 'Ts', 'Cr', 'Er', 'Re'],
                    help="Enabled augments (latest vut method only)", default=[], dest='clip_augments')
parser.add_argument("--clip_num_augs", type=int, help="Number of augmentations", default=16, dest='clip_num_augs')

# Content
parser.add_argument("--content_loss_coeff", type=float, help="Coefficient of Content Loss", default=1.0,
                    dest='content_loss_coeff')
# Overflow
parser.add_argument("--overflow_loss_weight", type=float, help="Coefficient of Overflow Loss", default=1.0,
                    dest='overflow_loss_weight')
# Regularization
parser.add_argument("--regularization_loss_weight", type=float, help="Coefficient of Latent Code Regularization Loss",
                    default=1e-2, dest='regularization_loss_weight')

# Only for Gaussian Pyramid
parser.add_argument("--max_level", type=int, help="Pyramid Levels",
                    default=5, dest='max_level')
parser.add_argument("--patch_size", type=int, help="Patch size for extracting features in the pyramid",
                    default=3, dest='patch_size')

# Optimization
parser.add_argument("--iterations", type=int, help="Number of iterations", default=2000, dest='max_iterations')
parser.add_argument("--save_every", type=int, help="Save image iterations", default=64, dest='save_every')
parser.add_argument("--batch_size", type=int, help="Batch size", default=4, dest='batch_size')
parser.add_argument("--lr", type=float, help="Learning rate", default=5e-3, dest='lr')
parser.add_argument("--lr_decay_step", nargs='+', action='append', type=int,
                    help="Specify the number of iterations for lr decay",
                    default=[], dest='lr_decay_step')
parser.add_argument("--optimizer", type=str, help="Optimizer",
                    choices=['Adam', 'AdamW', 'Adagrad', 'Adamax', 'DiffGrad', 'AdamP', 'RAdam', 'RMSprop'],
                    default='Adam', dest='optimizer')

parser.add_argument("--DEVICE", type=str, help="Cuda device to use", default="cuda:0", dest='DEVICE')

args = parser.parse_args()

DEVICE = torch.device(args.DEVICE if torch.cuda.is_available() else "cpu")

DynamicTextureLoss = Loss(args)

scale_factor = 1.0
c_out = 3

style_img = Image.open(args.style_path)
if args.texture_model == 'vgg':
    input_img_style, style_img_tensor = preprocess_style_image(style_img, model_type='vgg',
                                                               img_size=args.texture_img_size,
                                                               batch_size=args.batch_size)  # 0-1
    input_img_style = input_img_style.to(DEVICE)

nca_size_x, nca_size_y = int(args.nca_seed_size[0]), int(args.nca_seed_size[1])
condition_nca_size = (nca_size_x // 4, nca_size_y // 4)

scales = args.scales[0]  # [4, 1]
print(scales)
empty_str = ""
scale_str = f'{empty_str.join([str(x) for x in scales])}'
assert scales[-1] == 1
assert nca_size_x % scales[0] == 0
assert nca_size_y % scales[0] == 0

nca_perception_scales = args.nca_perception_scales[0]
print(nca_perception_scales)
empty_str = "x"
nca_perception_scales_str = f'pscale_[{empty_str.join([str(x) for x in nca_perception_scales])}]'
assert nca_perception_scales[0] == 0

scale_sizes = []
for scale in scales:
    scale_sizes.append((nca_size_x // scale, nca_size_y // scale))

'''Create the log folder'''
img_name = args.style_path.split('/')[-1].split('.')[0]
print(f"Target Texture: {img_name}")
dt = datetime.now()
timestamp = f"{dt.month}-{dt.day}-{dt.hour}-{dt.minute}"

output_dir = f'{args.output_dir}/{img_name}/{args.motion_field_name}/'

if not args.video_only:
    try:
        os.system(f"mkdir -p {output_dir}")
        os.system(f"rm -rf {output_dir}/*")
    except:
        pass
print('Create NCA model')

nca_min_steps, nca_max_steps = args.nca_step_range

nca_model = DyNCA(c_in=args.nca_c_in, c_out=c_out, fc_dim=args.nca_fc_dim,
                  seed_mode=args.nca_seed_mode,
                  pos_emb=args.nca_pos_emb, nca_pad_mode=args.nca_pad_mode,
                  perception_scales=nca_perception_scales,
                  device=DEVICE)
with torch.no_grad():
    nca_pool = nca_model.seed(args.nca_pool_size, size=(nca_size_x, nca_size_y))

param_n = sum(p.numel() for p in nca_model.parameters())
print('DyNCA param count:', param_n)

optimizer = torch.optim.Adam(nca_model.parameters(), lr=args.lr)

from utils.misc.video_utils import VideoWriter


def save_video(video_name, video_length, size_factor=1.0, step_n=8):
    fps = 30
    with VideoWriter(filename=f"{output_dir}/{video_name}.mp4", fps=fps, autoplay=False) as vid, torch.no_grad():
        h = nca_model.seed(1, size=(int(nca_size_x * size_factor), int(nca_size_y * size_factor)))
        for k in tqdm(range(int(video_length * fps)), desc="Making the video..."):
            nca_state, nca_feature = nca_model.forward_nsteps(h, step_n)

            z = nca_feature[-1]
            h = nca_state

            img = z.detach().cpu().numpy()[0]
            img = img.transpose(1, 2, 0)

            img = np.clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0
            vid.add(img)


args_log = copy.deepcopy(args.__dict__)
del args_log['DEVICE']
if ('target_motion_vec' in args_log):
    del args_log['target_motion_vec']
with open(f'{output_dir}/args.txt', 'w') as f:
    json.dump(args_log, f, indent=2)

if not args.video_only:
    pbar = tqdm(range(args.max_iterations), ncols=256)
else:
    pbar = tqdm(range(0))

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    args.lr_decay_step[0],
                                                    0.5)

input_dict = {}  # input dictionary for loss computing
input_dict['target_style_images'] = input_img_style  # 0,1

interval = args.motion_weight_change_interval

loss_log_dict = defaultdict(list)
new_size = (nca_size_x, nca_size_y)
for i in pbar:
    np.random.seed(i + 424)
    torch.manual_seed(i + 424)
    torch.cuda.manual_seed_all(i + 424)
    with torch.no_grad():
        batch_idx = np.random.choice(args.nca_pool_size, args.batch_size, replace=False)
        input_states = nca_pool[batch_idx]
        seed_injection = False
        if i % args.nca_inject_seed_step == 0:
            seed_injection = True
            seed_inject = nca_model.seed(1, size=(nca_size_x, nca_size_y))
            input_states[:1] = seed_inject[:1]

        '''Get the image before NCA iteration for computing optic flow'''
        input_states_before = input_states

        nca_states_before, nca_features_before = nca_model.forward_nsteps(input_states_before, step_n=1)
        z_before_nca = nca_features_before[-1]
        image_before_nca = z_before_nca

    step_n = np.random.randint(nca_min_steps, nca_max_steps)
    input_dict['step_n'] = step_n
    nca_states_after, nca_features_after = nca_model.forward_nsteps(input_states, step_n)

    z = nca_features_after[-1]
    generated_image = z
    with torch.no_grad():
        generated_image_vis = generated_image.clone()
        generated_image_vis = (generated_image_vis + 1.0) / 2.0

    image_after_nca = generated_image.clone()

    '''Construct input dictionary for loss computation'''
    input_dict['generated_images'] = generated_image
    input_dict['generated_image_before_nca'] = image_before_nca
    input_dict['generated_image_after_nca'] = image_after_nca

    input_dict['nca_state'] = nca_states_after

    DynamicTextureLoss.update_losses_to_apply(i)
    if (i + 1) % args.save_every == 0:
        batch_loss, batch_loss_log_dict, summary = DynamicTextureLoss(input_dict, return_summary=True)
    else:
        batch_loss, batch_loss_log_dict = DynamicTextureLoss(input_dict, return_summary=False)
        summary = {}

    for loss_name in batch_loss_log_dict:
        loss_log_dict[loss_name].append(batch_loss_log_dict[loss_name])

    if i % interval == 0:
        if i >= interval:
            print("Updating the motion loss weight")
            DynamicTextureLoss.set_loss_weight(loss_log_dict["texture"], "vector_field_motion")

    with torch.no_grad():
        batch_loss.backward()
        if torch.isnan(batch_loss):
            with open(f'{output_dir}/train_failed.txt', 'w') as f:
                f.write(f'Epochs {i}')
            print('Loss is NaN. Train Failed. Exit.')
            exit()

        for p_name, p in nca_model.named_parameters():
            p.grad /= (p.grad.norm() + 1e-8)  # normalize gradients
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        nca_pool[batch_idx] = nca_states_after

        if (i + 1) % args.save_every == 0:
            save_video("video_last", 3.0, size_factor=1.0, step_n=int(args.nca_base_num_steps))
            save_video("video_large_last", 3.0, size_factor=2.0, step_n=int(args.nca_base_num_steps))

            if 'motion-generated_video_flow' in summary:
                generated_flow_vis = summary['motion-generated_video_flow'] / 255.0
                save_train_image(generated_flow_vis[:4], f"{output_dir}/flow_gen{i}.jpg")

            if 'motion-generated_flow_vector_field' in summary:
                generated_flow_vector_field = summary['motion-generated_flow_vector_field']
                generated_flow_vector_field.save(f"{output_dir}/vec_field_gen_{i}.png")

            if 'motion-target_flow_vector_field' in summary:
                target_flow_vector_field = summary['motion-target_flow_vector_field']
                target_flow_vector_field.save(f"{output_dir}/vec_field_target.png")

            save_train_image(generated_image_vis.detach().cpu().numpy(), f"{output_dir}/step{i}.jpg")

            '''Dict: loss log, yscale to log (True/False), ylim (True/False)'''
            plot_log_dict = {}
            plot_log_dict['Overflow Loss'] = (loss_log_dict['overflow'], True, True)
            num_plots = 1
            if "texture" in loss_log_dict:
                num_plots += 1
                plot_log_dict['Texture Loss'] = (loss_log_dict['texture'], True, True)

            plot_train_log(plot_log_dict, num_plots, save_path=f"{output_dir}/losses.jpg")

            if "motion" in loss_log_dict:
                plot_log_dict = {}
                plot_log_dict['Motion Loss'] = (
                    loss_log_dict['motion'][args.motion_loss_iteration:], False, False)
                plot_log_dict['Motion Direction Loss'] = (loss_log_dict['motion-direction_loss'], False, False)
                plot_log_dict['Motion Strength Loss'] = (loss_log_dict['motion-strength_diff'], False, False)
                plot_train_log(plot_log_dict, 5, save_path=f"{output_dir}/losses_motion.jpg")

        if i % 5 == 0:
            display_dict = copy.deepcopy(batch_loss_log_dict)
            display_dict['lr'] = lr_scheduler.get_lr()[0]
            pbar.set_postfix(display_dict)

if not args.video_only:
    torch.save(nca_model, f"{output_dir}/model.pth")
else:
    layered_nca = torch.load(f"{output_dir}/model.pth")

save_video("video", args.video_length, size_factor=1.0, step_n=int(args.nca_base_num_steps))
save_video("video_large", args.video_length, size_factor=2.0, step_n=int(args.nca_base_num_steps))
