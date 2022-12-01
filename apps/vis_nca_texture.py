import os
import sys

sys.path.append("../")

import streamlit as st
import numpy as np
import torch
from PIL import Image
import models
from stqdm import stqdm

from utils.misc.video_utils import VideoWriter


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = torch.nn.functional.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x), indices


def z_to_img(model, z, quantize=False):
    if model:
        if quantize:
            z_q, _ = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight)
            z_q = z_q.movedim(3, 1)
        else:
            z_q = z
        return torch.clamp(model.decode(z_q), -1.0, 1.0)
    else:
        return torch.clamp(z, -1.0, 1.0)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache(ttl=None, allow_output_mutation=True, max_entries=3)
def load_pretrained_model(model_name):
    try:
        model = models.get_model(model_name, models_path="../pretrained_models", download=True)
    except:
        model = models.get_model(model_name, models_path="../pretrained_models", download=False)

    return model


experiments_dir = "../out/"


def choose_and_visualize():
    chosen_experiment = st.sidebar.selectbox("Choose the experiments you want to visualize",
                                             os.listdir(experiments_dir))
    last_path = os.path.join(experiments_dir, chosen_experiment)
    last_file_list = os.listdir(last_path)
    last_file_list.sort()
    chosen_tag = st.sidebar.selectbox("Choose the VQAE decoder", last_file_list)
    last_path = os.path.join(last_path, chosen_tag)
    last_file_list = os.listdir(last_path)
    last_file_list.sort()
    chosen_texture = st.sidebar.selectbox("Choose the Texture", last_file_list)
    last_path = os.path.join(last_path, chosen_texture)
    last_file_list = os.listdir(last_path)
    last_file_list.sort()
    chosen_text_prompt = st.sidebar.selectbox("Choose the texture model with the loss", last_file_list)

    last_path = os.path.join(last_path, chosen_text_prompt)
    # avaliable_dates = os.listdir(last_path)
    # avaliable_dates = list(reversed(sorted(avaliable_dates, key=lambda x: datetime.datetime.strptime(x, '%m-%d-%H-%M'))))
    # last_date = avaliable_dates[0]

    # chosen_date = st.sidebar.selectbox("Choose the experiment date tag", avaliable_dates)
    # last_path = os.path.join(last_path, chosen_date)

    available_files = os.listdir(last_path)
    available_files.sort()

    config_txt = open(os.path.join(last_path, 'args.txt')).read()
    st.json(config_txt, expanded=False)
    
    if(os.path.exists(os.path.join(last_path, 'train_failed.txt'))):
        failed_text = open(os.path.join(last_path, 'train_failed.txt')).readlines()
        st.write(failed_text[0])
    
    if(os.path.exists(os.path.join(last_path, 'median_loss.txt'))):
        loss_val = open(os.path.join(last_path, 'median_loss.txt')).readlines()
        st.write(loss_val[0])

    def display_img(filename, caption):
        if filename in available_files:
            loss_img_file = open(os.path.join(last_path, filename), 'rb')
            loss_img = Image.open(loss_img_file)
            st.image(loss_img, caption=caption)
    display_img('select_frame.png', 'Frame For Texture Fitting')
    display_img('losses.jpg', 'Loss values during optimization')
    display_img("losses_motion.jpg", 'Motion Loss values during optimization')
    display_img('losses-adv.jpg', 'Loss values during adversarial reprogramming')
    display_img('losses_motion_texture.jpg', 'Motion Texture Loss')
    display_img('loss_distribution.jpg', 'Loss Distribution')

    max_step = max([int(s[4:-4]) for s in os.listdir(last_path) if "step" in s and "adv" not in s])
    display_img(f'step{max_step}.jpg', 'Generated Images in the last optimization step')
    display_img(f'step-adv{max_step}.jpg', 'Generated Images in the last adversarial reprogramming step')
    display_img(f'flow_gen{max_step}.jpg', 'Optic flow between generated images')
    display_img(f'flow_target{max_step}.jpg', 'Optic flow between target videos')
    display_img(f'vec_gen{max_step}.png', 'Optic flow between generated images')
    display_img(f'vec_target{max_step}.png', 'Optic flow between target videos')
    display_img('hist.png', 'Hist of value in the gradient map')

    def display_video(video_name):
        if (os.path.exists(os.path.join(last_path, video_name))):
            video_file = open(os.path.join(last_path, video_name), 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
    display_video('video_train.mp4')
    display_video('video.mp4')
    display_video('video_large.mp4')
    
    if(os.path.exists(os.path.join(last_path, 'final_loss.txt'))):
        loss_val = open(os.path.join(last_path, 'final_loss.txt')).readlines()
        st.write(loss_val[0])
    
    if(os.path.exists(os.path.join(last_path, 'final_loss_test.txt'))):
        loss_val = open(os.path.join(last_path, 'final_loss_test.txt')).readlines()
        st.write(loss_val[0])
    
    # display_video('video_analyze.mp4')
    # display_video('video_analyze2.mp4')
    # display_video('video_adv.mp4')
    # display_video('video_add.mp4')
    # display_video('video_rotate.mp4')

    delta_T = float(st.sidebar.slider(f'Delta T', 0.1, 2.0, 1.0))
    resolution = 2 ** int(st.sidebar.text_input('Log Image Resolution', value='6'))
    num_steps_per_frame = int(st.sidebar.text_input('Number of steps per frame', value='1'))
    video_length = int(st.sidebar.text_input('Video Length in seconds', value='10'))

    if st.sidebar.button('Generate New Video (Not working yet)'):
        if chosen_vq_model != "no_vqae":
            vqae_model = load_pretrained_model(chosen_vq_model)
            vqae_model = vqae_model.to(DEVICE)
        else:
            vqae_model = None
        nca_model = torch.load(os.path.join(last_path, 'model.pth')).to(DEVICE)
        # print(nca_model.trainable_kernel)

        with VideoWriter(filename=f"tmp.mp4", autoplay=False) as vid, torch.no_grad():
            # nca_model.random_seed = 3
            h = nca_model.seed(1, size=(resolution, resolution))
            for k in stqdm(range(int(video_length * 30)), desc="Making the video..."):
                step_n = min(2 ** (k // 30), 16)
                for i in range(num_steps_per_frame):
                    h[:] = nca_model(h, update_rate=0.5, delta_T=delta_T)

                z = nca_model.to_feature_map(h)
                img = z_to_img(vqae_model, z, quantize=False).detach().cpu().numpy()[0]
                img = img.transpose(1, 2, 0)
                img = np.clip(img, -1.0, 1.0)
                img = (img + 1) / 2.0
                vid.add(img)

        video_file = open('tmp.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)


choose_and_visualize()