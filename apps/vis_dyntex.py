import os
import sys

sys.path.append("../")

import streamlit as st

import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.misc.video_utils import VideoWriter



dyntex_dataset, collate_fn = dataset.get_dataset("DynTex", "/scratch/Ehsan/data/DynTex/")

dataset_len = len(dyntex_dataset)
sample_idx = int(st.sidebar.text_input(f'Select the sample Index from 0 to {dataset_len}', value='0'))

texture_video, metadata = dyntex_dataset[sample_idx]

texture_video = texture_video.cpu().numpy()
# texture_img = texture_img.cpu().numpy().transpose(1, 2, 0)
with VideoWriter(filename=f"tmp.mp4", autoplay=False) as vid:
    for i in range(texture_video.shape[1]):
        x = texture_video[:, i].transpose(1, 2, 0)
        vid.add(x)

        
search_phrase = st.sidebar.text_input("Search phrase")
if st.sidebar.button("search"):
    ids = dyntex_dataset.search(search_phrase)
    st.text(ids)

video_file = open('tmp.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

# st.image(texture_img)
st.json(metadata)