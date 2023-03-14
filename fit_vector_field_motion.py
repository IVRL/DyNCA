import os
import warnings

import json
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy

from collections import defaultdict

from models.dynca import DyNCA

from utils.misc.display_utils import plot_train_log, save_train_image
from utils.misc.preprocess_texture import preprocess_style_image
from utils.misc.video_utils import VideoWriter

from utils.loss.loss import Loss

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
torch.backends.cudnn.deterministic = True

import argparse

parser = argparse.ArgumentParser(description='DyNCA - Dynamic Texture Synthesis from Motion Vector Field')

# Add the arguments
parser.add_argument("--motion_img_size", nargs=2, type=int,
                    help="Image size (width height) to compute motion loss | default = (128, 128)",
                    default=[128, 128], dest='motion_img_size')
parser.add_argument("--img_size", nargs=2, type=int,
                    help="Seed size of NCA model (width, height) | default = (128, 128)",
                    default=[128, 128], dest='img_size')

parser.add_argument("--output_dir", type=str, help="Output directory", default="out/VectorFieldMotion/",
                    dest='output_dir')
parser.add_argument("--video_length", type=float, help="Video length in seconds (not interpolated)", default=10,
                    dest='video_length')
parser.add_argument("--video_only", action='store_true', help="Only generate video using pretrained model",
                    dest='video_only')

# Target
parser.add_argument("--target_appearance_path", type=str, help="Path to the target appearance image",
                    default='data/VectorFieldMotion/Appearance/bubbly_0101.jpg',
                    dest='target_appearance_path')

# NCA
parser.add_argument("--nca_pool_size", type=int, help="Number of elements in the NCA pool", default=256,
                    dest='nca_pool_size')
parser.add_argument("--nca_step_range", nargs=2, type=int, help="Range of steps to apply NCA (32, 128)",
                    default=[32, 128], dest='nca_step_range')
parser.add_argument("--nca_inject_seed_step", type=int, help="Inject seed every time after this many iterations",
                    default=8, dest='nca_inject_seed_step')
parser.add_argument("--nca_channels", type=int, help="Number of Channels in the NCA model", default=12, dest='nca_c_in')
parser.add_argument("--nca_fc_dim", type=int, help="FC layer dimension", default=96, dest='nca_fc_dim')
parser.add_argument("--nca_seed_mode", type=str, help="Seed mode of the NCA", default='zeros',
                    choices=DyNCA.SEED_MODES, dest='nca_seed_mode')
parser.add_argument("--nca_padding_mode", type=str, default='replicate',
                    help="Padding mode when NCA cells are perceiving",
                    choices=['constant', 'reflect', 'replicate', 'circular'],
                    dest='nca_padding_mode')
parser.add_argument("--nca_pos_emb", type=str, default='CPE', choices=['None', 'CPE'],
                    help="The positional embedding mode to use. CPE (Cartesian), or None",
                    dest='nca_pos_emb')
parser.add_argument("--nca_perception_scales", nargs='+', action='append', type=int,
                    help="Specify the scales at which the NCA perception will be performed.",
                    default=[], dest='nca_perception_scales')

# Loss Function
# Appearance
parser.add_argument("--appearance_loss_weight", type=float,
                    help="Coefficient of Loss used for Appearance Loss", default=1.0,
                    dest='appearance_loss_weight')
parser.add_argument("--appearance_loss_type", type=str,
                    help="The method to compute appearance loss. Sliced W-distance , OT (Optimal transport), Gram",
                    choices=["SlW", "OT", "Gram"],
                    default="OT",
                    dest='appearance_loss_type')

# Vector Field Motion
parser.add_argument("--motion_loss_weight", type=float, help="Coefficient of Motion Loss", default=4.0,
                    dest='vector_field_motion_loss_weight')
parser.add_argument("--motion_strength_weight", type=float, help="Coefficient of Motion enhancing loss", default=15.0,
                    dest='motion_strength_weight')
parser.add_argument("--motion_direction_weight", type=float, help="Coefficient of direction indicating loss",
                    default=10.0,
                    dest='motion_direction_weight')
parser.add_argument("--motion_weight_change_interval", type=int,
                    help="Interval of iterations for changing the motion loss weight. ",
                    default=500,
                    dest='motion_weight_change_interval')
parser.add_argument("--motion_vector_field_name", type=str,
                    help="Name of the motion vector field to be used", default="circular",
                    dest='motion_vector_field_name')
parser.add_argument("--motion_model_name", type=str, default='two_stream_dynamic',
                    help="Optic Flow computing model. Default is two_stream_dynamic. ",
                    dest='motion_model_name')

parser.add_argument("--nca_base_num_steps", type=float,
                    help="Number of NCA steps to normalize the magnitude of the optic flow. This refers to the parameter T in the paper",
                    default=24.0,
                    dest='nca_base_num_steps')

# Overflow
parser.add_argument("--overflow_loss_weight", type=float, help="Coefficient of Overflow Loss", default=100.0,
                    dest='overflow_loss_weight')

# Optimization
parser.add_argument("--iterations", type=int, help="Number of iterations", default=2000, dest='max_iterations')
parser.add_argument("--save_every", type=int, help="Save image iterations", default=200, dest='save_every')
parser.add_argument("--batch_size", type=int, help="Batch size", default=4, dest='batch_size')
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3, dest='lr')
parser.add_argument("--lr_decay_step", nargs='+', action='append', type=int,
                    help="Specify the number of iterations for lr decay",
                    default=[], dest='lr_decay_step')

parser.add_argument("--DEVICE", type=str, help="Cuda device to use", default="cuda:0", dest='DEVICE')

args = parser.parse_args()

DEVICE = torch.device(args.DEVICE if torch.cuda.is_available() else "cpu")

DynamicTextureLoss = Loss(args)

style_img = Image.open(args.target_appearance_path)
target_appearance_img = preprocess_style_image(style_img, model_type='vgg',
                                               img_size=args.img_size,
                                               batch_size=args.batch_size) * 2.0 - 1.0  # [-1.0, 1.0]
target_appearance_img = target_appearance_img.to(DEVICE)

nca_size_x, nca_size_y = int(args.img_size[0]), int(args.img_size[1])

try:
    nca_perception_scales = args.nca_perception_scales[0]
except:
    nca_perception_scales = [0]
assert nca_perception_scales[0] == 0

'''Create the log folder'''
img_name = args.target_appearance_path.split('/')[-1].split('.')[0]
print(f"Target Appearance: {img_name}")

output_dir = f'{args.output_dir}/{img_name}/{args.motion_vector_field_name}/'

if not args.video_only:
    try:
        os.system(f"mkdir -p {output_dir}")
        os.system(f"rm -rf {output_dir}/*")
    except:
        pass

print('Creating DyNCA model')

nca_min_steps, nca_max_steps = args.nca_step_range

nca_model = DyNCA(c_in=args.nca_c_in, c_out=3, fc_dim=args.nca_fc_dim,
                  seed_mode=args.nca_seed_mode,
                  pos_emb=args.nca_pos_emb, padding_mode=args.nca_padding_mode,
                  perception_scales=nca_perception_scales,
                  device=DEVICE)
with torch.no_grad():
    nca_pool = nca_model.seed(args.nca_pool_size, size=(nca_size_x, nca_size_y))

param_n = sum(p.numel() for p in nca_model.parameters())
print('DyNCA param count:', param_n)

optimizer = torch.optim.Adam(nca_model.parameters(), lr=args.lr)


def save_video(video_name, video_length, size_factor=1.0, step_n=8):
    fps = 30
    with VideoWriter(filename=f"{output_dir}/{video_name}.mp4", fps=fps, autoplay=False) as vid, torch.no_grad():
        h = nca_model.seed(1, size=(int(nca_size_x * size_factor), int(nca_size_y * size_factor)))
        for k in tqdm(range(int(video_length * fps)), desc="Making the video..."):
            nca_state, nca_feature = nca_model.forward_nsteps(h, step_n)

            z = nca_feature
            h = nca_state

            img = z.detach().cpu().numpy()[0]
            img = img.transpose(1, 2, 0)

            img = np.clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0
            vid.add(img)


args_log = copy.deepcopy(args.__dict__)
del args_log['DEVICE']
if 'target_motion_vec' in args_log:
    del args_log['target_motion_vec']

with open(f'{output_dir}/args.txt', 'w') as f:
    json.dump(args_log, f, indent=2)

if not args.video_only:
    pbar = tqdm(range(args.max_iterations), ncols=256)
else:
    pbar = tqdm(range(0))

if len(args.lr_decay_step) == 0:
    args.lr_decay_step = [[1000, 2000]]

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    args.lr_decay_step[0],
                                                    0.5)

input_dict = {}  # input dictionary for computing the loss functions
input_dict['target_image_list'] = [target_appearance_img]  # 0,1

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

        nca_states_before, nca_features_before = nca_model.forward_nsteps(input_states, step_n=1)
        z_before_nca = nca_features_before
        image_before_nca = z_before_nca

    step_n = np.random.randint(nca_min_steps, nca_max_steps)
    input_dict['step_n'] = step_n
    nca_states_after, nca_features_after = nca_model.forward_nsteps(input_states, step_n)

    z = nca_features_after
    generated_image = z
    with torch.no_grad():
        generated_image_vis = generated_image.clone()
        generated_image_vis = (generated_image_vis + 1.0) / 2.0

    image_after_nca = generated_image.clone()

    '''Construct input dictionary for loss computation'''
    input_dict['generated_image_list'] = [generated_image]
    input_dict['generated_image_before_nca'] = image_before_nca
    input_dict['generated_image_after_nca'] = image_after_nca

    input_dict['nca_state'] = nca_states_after

    if (i + 1) % args.save_every == 0:
        batch_loss, batch_loss_log_dict, summary = DynamicTextureLoss(input_dict, return_summary=True)
    else:
        batch_loss, batch_loss_log_dict, _ = DynamicTextureLoss(input_dict, return_summary=False)
        summary = {}

    for loss_name in batch_loss_log_dict:
        loss_log_dict[loss_name].append(batch_loss_log_dict[loss_name])

    if i % interval == 0 and i > 0:
        print("Updating the motion loss weight")
        DynamicTextureLoss.set_loss_weight(loss_log_dict["appearance"], "vector_field_motion")

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

            if 'vector_field_motion-generated_video_flow' in summary:
                generated_flow_vis = summary['vector_field_motion-generated_video_flow'] / 255.0
                save_train_image(generated_flow_vis[:4], f"{output_dir}/flow_gen{i}.jpg")

            if 'vector_field_motion-generated_flow_vector_field' in summary:
                generated_flow_vector_field = summary['vector_field_motion-generated_flow_vector_field']
                generated_flow_vector_field.save(f"{output_dir}/vec_field_gen{i}.png")

            if 'vector_field_motion-target_flow_vector_field' in summary:
                target_flow_vector_field = summary['vector_field_motion-target_flow_vector_field']
                target_flow_vector_field.save(f"{output_dir}/vec_field_target.png")

            save_train_image(generated_image_vis.detach().cpu().numpy(), f"{output_dir}/step{i}.jpg")

            '''Dict: loss log, yscale to log (True/False), ylim (True/False)'''
            plot_log_dict = {}
            plot_log_dict['Overflow Loss'] = (loss_log_dict['overflow'], True, True)
            num_plots = 1
            if "appearance" in loss_log_dict:
                num_plots += 1
                plot_log_dict['Texture Loss'] = (loss_log_dict['appearance'], True, True)

            plot_train_log(plot_log_dict, num_plots, save_path=f"{output_dir}/losses.jpg")

            if "vector_field_motion" in loss_log_dict:
                plot_log_dict = {}
                plot_log_dict['Motion Loss'] = (
                    loss_log_dict['vector_field_motion'], False, False)
                plot_log_dict['Motion Direction Loss'] = (
                    loss_log_dict['vector_field_motion-direction'], False, False)
                plot_log_dict['Motion Strength Loss'] = (
                    loss_log_dict['vector_field_motion-strength'], False, False)
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
