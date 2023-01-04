import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageSequence
import cv2
import random
import copy
def preprocess_style_image(style_img, model_type = 'vgg', img_size = [128, 128], batch_size = 4):
    if(model_type == 'vgg'):
        w, h = style_img.size
        if(w == h):
            style_img = style_img.resize((img_size[0],img_size[1]))
            style_img = style_img.convert('RGB')
            # with torch.no_grad():
            #     style_img_tensor = transforms.ToTensor()(style_img).unsqueeze(0)
        else:
            style_img = style_img.convert('RGB')
            style_img = np.array(style_img)
            h, w, _ = style_img.shape
            cut_pixel = abs(w - h) // 2
            if(w > h):
                style_img = style_img[:, cut_pixel:w-cut_pixel, :]
            else:
                style_img = style_img[cut_pixel:h-cut_pixel, :, :]
            style_img = Image.fromarray(style_img.astype(np.uint8))
            style_img = style_img.resize((img_size[0],img_size[1]))
        style_img = np.float32(style_img) / 255.0
        style_img = torch.as_tensor(style_img)
        style_img = style_img[None,...]
        input_img_style = style_img.permute(0, 3, 1, 2)
        input_img_style = input_img_style.repeat(batch_size, 1, 1, 1)
        return input_img_style#, style_img_tensor
    
def preprocess_video(video_path, img_size=[128, 128], normalRGB = False):
    if ('.gif' in video_path):
        gif_video = Image.open(video_path)
        train_image_seq = []
        index = 0
        for frame in ImageSequence.Iterator(gif_video):
            cur_frame_tensor = preprocess_style_image(frame, 'vgg', img_size)
            if(normalRGB == False):
                cur_frame_tensor = cur_frame_tensor * 2.0 - 1.0
                
            train_image_seq.append(cur_frame_tensor)
            index += 1
        train_image_seq = torch.stack(train_image_seq, dim=2)[0] # Output shape is [C, T, H, W]
        # print(f'Total Training Frames: {index}')
        return train_image_seq
    elif('.avi' in video_path or '.mp4' in video_path):
        cap = cv2.VideoCapture(video_path)
        train_image_seq = []
        index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
#                 print(f'Total Training Frames: {index}')
                break
            index += 1
#             if(index == 50):
#                 break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR->RGB
            frame = Image.fromarray(frame.astype(np.uint8)).convert('RGB')

            cur_frame_tensor = preprocess_style_image(frame, 'vgg', img_size)
            if(normalRGB == False):
                cur_frame_tensor = cur_frame_tensor * 2.0 - 1.0
                
            train_image_seq.append(cur_frame_tensor)
        train_image_seq = torch.stack(train_image_seq, dim=2)[0]
        
        cap.release()
        cv2.destroyAllWindows()
        return train_image_seq


def select_frame(args, image_seq, vgg_model):
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    image_seq_vgg = (image_seq + 1.0) / 2.0
    feature_map = get_middle_feature_vgg(args, image_seq_vgg, vgg_model)[-2:-1]
    # feature_norm = [torch.norm(x.reshape(len(image_seq_vgg), -1), dim = 1).reshape(len(image_seq_vgg), 1, 1, 1) for x in feature_map]
    # feature_map = [x / y for x,y in zip(feature_map, feature_norm)]
    
    
    avg_feature_map = [torch.mean(x, dim = 0) for x in feature_map]
    min_dist_idx_list = []
    dist_array = np.zeros((len(feature_map), len(image_seq_vgg)))
    for i in range(len(feature_map)):
        feature_map_single = feature_map[i]
        avg_feature_map_single = avg_feature_map[i]
        dist_norm = [torch.mean(torch.norm(x - avg_feature_map_single)).item() for x in feature_map_single]
        # c = feature_map_single.shape[1]
        # dist_norm = [torch.mean(cos_sim(x.reshape(c, -1), avg_feature_map_single.reshape(c, -1))).item() for x in feature_map_single]
        frame_idx = np.argmin(dist_norm)
        min_dist_idx_list.append(frame_idx)
        dist_array[i] = np.array(dist_norm)
    # print(min_dist_idx_list)
    # print(dist_array)
    dist_mean = np.mean(dist_array, axis = 0)
    # print(dist_mean)
    # print(np.argmin(dist_mean))
    frame_idx = np.argmin(dist_mean)
    return frame_idx

def get_train_image_seq(args, **kwargs):
    if('.png' in args.target_appearance_path or '.jpg' in args.target_appearance_path or '.jpeg' in args.target_appearance_path):
        style_img = Image.open(args.target_appearance_path)
        train_image_seq_texture = preprocess_style_image(style_img, model_type = 'vgg', img_size=args.img_size)
        train_image_seq_texture = train_image_seq_texture[0:1].to(args.DEVICE) # 1, C, H, W
        train_image_seq_texture = (train_image_seq_texture * 2.0) - 1.0
        frame_idx_texture = 0
        train_image_texture = copy.deepcopy(train_image_seq_texture[frame_idx_texture])
        train_image_texture_save = transforms.ToPILImage()((train_image_texture + 1.0) / 2.0)
    else:
        flow_func = kwargs.get('flow_func', None)
        train_image_seq_sort = preprocess_video(args.target_dynamics_path, img_size=(256,256))
        train_image_seq_sort = train_image_seq_sort.permute(1, 0, 2, 3).to(args.DEVICE)
        video_length = len(train_image_seq_sort)
        frame_weight_list = []
        with torch.no_grad():
            for idx in range(video_length - 1):
                image1 = train_image_seq_sort[idx:idx+1]
                image2 = train_image_seq_sort[idx+1:idx+2]

                _, flow = flow_func(image1, image2, size=(256,256))
                motion_strength = torch.mean(torch.norm(flow, dim = 1)).item()
                frame_weight_list.append(motion_strength)
        total_strength = sum(frame_weight_list)
        frame_weight_list = [x / total_strength for x in frame_weight_list]
        train_image_seq_texture = preprocess_video(args.target_appearance_path, img_size=args.img_size)
        train_image_seq_texture = train_image_seq_texture.permute(1, 0, 2, 3).to(args.DEVICE) # T, C, H, W
        texture_video_length = len(train_image_seq_texture)
        frame_idx_texture = np.argmax(frame_weight_list)
        if(frame_idx_texture >= texture_video_length):
            frame_idx_texture = random.randint(0, texture_video_length - 1)
        train_image_texture = copy.deepcopy(train_image_seq_texture[frame_idx_texture])
        train_image_texture_save = transforms.ToPILImage()((train_image_texture + 1.0) / 2.0)
    return train_image_seq_texture,train_image_texture,train_image_texture_save,frame_idx_texture
    
def get_middle_feature_vgg(args, imgs, vgg_model, flatten=False, include_image_as_feat = False):
    size = args.img_size
    DEVICE = args.DEVICE
    img_shape = imgs.shape[2]
    # if (img_shape == size[0]):
    #     pass
    # else:
    #     imgs = TF.resize(imgs, size)
    style_layers = [1, 6, 11, 18, 25]  # 1, 6, 11, 18, 25
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(DEVICE)
    x = (imgs - mean) / std
    b, c, h, w = x.shape
    if(include_image_as_feat):
        features = [x.reshape(b, c, h*w)]
    else:
        features = []
    for i, layer in enumerate(vgg_model[:max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            if flatten:
                features.append(x.reshape(b, c, h * w))
            else:
                features.append(x)
    return features
