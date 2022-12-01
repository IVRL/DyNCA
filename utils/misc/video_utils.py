import os
import torch
import numpy as np
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm

os.environ['FFMPEG_BINARY'] = 'ffmpeg'


class VideoWriter:
    def __init__(self, filename='tmp.mp4', fps=30.0, autoplay=False, **kw):
        self.writer = None
        self.autoplay = autoplay
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.autoplay:
            self.show()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))

def synthesize_video(args, nca_model, video_length, output_dir, train_image_seq_texture, train_image_seq, video_name = 'video', nca_step = 32, record_loss = False, loss_class = None, seed_size = [256,256], fps = 25):
    motion_video_length, texture_videl_length = len(train_image_seq), len(train_image_seq_texture)
    with VideoWriter(filename=f"{output_dir}/{video_name}.mp4", fps=fps, autoplay=False) as vid, torch.no_grad():
        h = nca_model.seed(1, size=seed_size)
        if(record_loss):
            assert loss_class is not None
            prev_z = None
            total_motion_texture_loss_avg = 0.0
            total_texture_loss_avg = 0.0
        for k in tqdm(range(int(video_length)), desc="Making the video..."):
            step_n = nca_step
            nca_state, nca_feature = nca_model.forward_nsteps(h, step_n)

            z = nca_feature
            
            if(record_loss):
                input_dict = {}
                cur_motion_texture_loss_avg = 0.0
                cur_texture_loss_avg = 0.0

                if(prev_z is None):
                    prev_z = z
                else:
                    generated_image_list = [prev_z, z]
                    input_dict['generated_image_list'] = [generated_image_list[-1]]
                    input_dict['generated_image_list_motion'] = generated_image_list
                    for j in range(texture_videl_length):
                        '''Compute Texture loss between current generated image and all texture frames'''
                        target_image_list = []
                        target_image_list.append(train_image_seq_texture[j:j+1])
                        input_dict['target_image_list'] = target_image_list
                        texture_loss,_,_ = loss_class.loss_mapper['texture'](input_dict, return_summary = False)
                        cur_texture_loss_avg += texture_loss.item()
                    cur_texture_loss_avg /= texture_videl_length

                    for j in range(motion_video_length - 1):
                        target_motion_image_list = []
                        target_motion_image_list.append(train_image_seq[j:j+1])
                        target_motion_image_list.append(train_image_seq[j+1:j+2])
                        input_dict['target_motion_image_list'] = target_motion_image_list
                        motion_texture_loss,_,_ = loss_class.loss_mapper['motion_texture'](input_dict, return_summary = False)
                        cur_motion_texture_loss_avg += motion_texture_loss.item()
                    cur_motion_texture_loss_avg /= (motion_video_length - 1)

                    total_texture_loss_avg += cur_texture_loss_avg
                    total_motion_texture_loss_avg += cur_motion_texture_loss_avg

                    prev_z = z

            h = nca_state

            img = z.detach().cpu().numpy()[0]
            img = img.transpose(1, 2, 0)

            img = np.clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0
            vid.add(img)
        if(record_loss):
            total_texture_loss_avg /= float(args.video_length * 40)
            total_motion_texture_loss_avg /= float(args.video_length * 40)
            with open(f'{output_dir}/final_loss_test.txt', 'w') as f:
                f.write(f'{total_texture_loss_avg, total_motion_texture_loss_avg}') 
