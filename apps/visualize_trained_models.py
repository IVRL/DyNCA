import os
import sys

sys.path.append("../")

import streamlit as st
import numpy as np
import torch
from PIL import Image
import models
from stqdm import stqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiments_dir = "../out/"


def choose_and_visualize():
    chosen_experiment = st.sidebar.selectbox("Choose the experiments you want to visualize",
                                             os.listdir(experiments_dir))
    last_path = os.path.join(experiments_dir, chosen_experiment)
    last_file_list = os.listdir(last_path)
    last_file_list.sort()
    chosen_texture = st.sidebar.selectbox("Choose the Appearance Target", last_file_list)
    last_path = os.path.join(last_path, chosen_texture)
    last_file_list = os.listdir(last_path)
    last_file_list.sort()
    chosen_motion = st.sidebar.selectbox("Choose the Motion Target", last_file_list)
    last_path = os.path.join(last_path, chosen_motion)

    available_files = os.listdir(last_path)
    available_files.sort()

    config_txt = open(os.path.join(last_path, 'args.txt')).read()
    st.json(config_txt, expanded=False)

    def display_img(filename, caption):
        if filename in available_files:
            loss_img_file = open(os.path.join(last_path, filename), 'rb')
            loss_img = Image.open(loss_img_file)
            st.image(loss_img, caption=caption)
    display_img('select_frame.png', 'Target Appearance')
    display_img('losses.jpg', 'Loss values during optimization')
    display_img('losses_video_motion.jpg', 'Motion Loss')
    display_img('losses_motion.jpg', 'Motion Loss')

    max_step = max([int(s[4:-4]) for s in os.listdir(last_path) if "step" in s and "adv" not in s])
    display_img(f'step{max_step}.jpg', 'Generated Images in the last optimization step')
    display_img(f'flow_gen{max_step}.jpg', 'Optic flow between generated images')
    display_img(f'flow_target{max_step}.jpg', 'Optic flow between target videos')
    
    display_img(f'vec_field_gen{max_step}.png', 'Synthesized Vector Field')
    display_img(f'vec_field_target.png', 'Target Vector Field')

    def display_video(video_name):
        if (os.path.exists(os.path.join(last_path, video_name))):
            video_file = open(os.path.join(last_path, video_name), 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
    display_video('video_last.mp4')
    display_video('video_large_last.mp4')        
    
    display_video('video.mp4')
    display_video('video_large.mp4')


choose_and_visualize()