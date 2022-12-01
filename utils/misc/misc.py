import numpy as np
import random

def get_start_frame_idx(video_length, image_list_length):
    idx_list = list(range(video_length - 1))
    frame_weight_list = np.ones((len(idx_list)))
    idx_vid = random.choices(idx_list)[0]
    start_frame_idx = idx_vid
    frame_num_left = image_list_length - 2
    left_add = 1
    right_add = 1
    while frame_num_left > 0:
        left_expand = idx_vid - left_add
        right_expand = idx_vid + right_add + 1 # This plus one is the next frame of the choosing beginning frame
        cur_weight = [0, 0]
        if(left_expand < 0):
            cur_weight[1] = 1 # 100% to pick right
        elif(right_expand > video_length - 1):
            cur_weight[0] = 1 # 100% to pick left
        else:
            cur_weight[0] = frame_weight_list[left_expand]
            cur_weight[1] = frame_weight_list[right_expand - 1]
        cur_add_frame = random.choices([0, 1], weights = cur_weight)[0] # 0 is left, 1 is right
        if(cur_add_frame == 0):
            left_add += 1
            start_frame_idx -= 1
        else:
            right_add += 1
        frame_num_left -= 1
    return start_frame_idx

def save_summary(summary, save_func, output_dir, i = 0):
    if(summary is not None):
        generated_flow_vis = summary['video_motion-generated_video_flow'] / 255.0
        save_func(generated_flow_vis, f"{output_dir}/flow_gen{i}.jpg")

        target_flow_vis = summary['video_motion-target_video_flow'] / 255.0
        save_func(target_flow_vis, f"{output_dir}/flow_target{i}.jpg")

        generated_vec_vis = summary['video_motion-generated_video_vec']
        save_func(generated_vec_vis, f"{output_dir}/vec_gen{i}.png")

        target_vec_vis = summary['video_motion-target_video_vec']
        save_func(target_vec_vis, f"{output_dir}/vec_target{i}.png")