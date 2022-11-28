import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms 

from segformer import segformer_mit_b3



def preprocess_image(image_path, tf, patch_size):
    '''preprocess image for visualization'''
    # read image -> convert to RGB -> torch Tensor
    rgb_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = tf(rgb_img)
    _, image_height, image_width = img.shape
    
    # make the image divisible by the patch size
    w, h = image_width - image_width % patch_size, image_height - image_height % patch_size
    img = img[:, :h, :w].unsqueeze(0)   # torch.Size([1, 3, h, w])
    
    w_featmap = img.shape[-1] // patch_size
    h_featmap = img.shape[-2] // patch_size
    return rgb_img, img, w_featmap, h_featmap

def preprocess_image2(rgb_img, tf, patch_size):
    resize_img = cv2.resize(rgb_img, (1248, 384), interpolation=cv2.INTER_AREA) # for input size = (1241, 376)
    img = tf(resize_img)
    _, image_height, image_width = img.shape
    
    # make the image divisible by the patch size
    w, h = image_width - image_width % patch_size, image_height - image_height % patch_size
    img = img[:, :h, :w].unsqueeze(0)   # torch.Size([1, 3, h, w])

    w_featmap = img.shape[-1] // patch_size
    h_featmap = img.shape[-2] // patch_size
    return resize_img, img, w_featmap, h_featmap


def calculate_single_stage_attention(model, img, stage_num, targetHeight, targetWidth, stage_scale, mode='bilinear'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    stages_data = model.get_attention_outputs(img.to(device))
    stage_attn = stages_data[stage_num]['attn']

    stage_heads = stage_attn.shape[1]

    # keep the output patch attention and reshape to feature map size
    stage_h, stage_w = int(targetHeight / stage_scale[stage_num]), int(targetWidth / stage_scale[stage_num])
    stage_attn = stage_attn[0, :, :, 0].reshape(stage_heads, stage_h, stage_w)

    # resize back to original image size
    stage_attn = F.interpolate(stage_attn.unsqueeze(0), size=(targetHeight, targetWidth), mode=mode)[0].detach().cpu().numpy()
    return stage_attn

def get_single_stage_attention(image_path, model, transform, patch_size, num_stages, targetHeight, targetWidth, stage_scale, mode = 'bilinear'):
    rgb_img, img, _, _ = preprocess_image(image_path, transform, patch_size)
    attentions = calculate_single_stage_attention(model, img, num_stages, targetHeight, targetWidth, stage_scale, mode='bilinear')
    return rgb_img, attentions

def get_single_stage_attention2(image, model, transform, patch_size, num_stages, targetHeight, targetWidth, stage_scale, mode = 'bilinear'):
    rgb_img, img, _, _ = preprocess_image2(image, transform, patch_size)
    attentions = calculate_single_stage_attention(model, img, num_stages, targetHeight, targetWidth, stage_scale, mode='bilinear')
    return rgb_img, attentions

def plot_and_save_image(images_path, model, transform, patch_size, targetHeight, targetWidth, stage_scale, show_comb):
    # set plot titles
    stage_heads = [1, 2, 5, 8]
    titles = []
    for stage_index, stage_nh in enumerate(stage_heads):
        titles.extend([f"STAGE_{stage_index+1}_HEAD_{x+1}" for x in range(stage_nh)])
    
    # plot the attention images
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 4}
    plt.rc('font', **font)
    plt.rcParams['text.color'] = 'white'
    fig, axes = plt.subplots(3,3, figsize=(15.5,8))
    axes = axes.flatten()
    fig.tight_layout()

    for image_path in tqdm(images_path):
        image_name = image_path.split(os.sep)[-1].split('.')[0]
        
        # get stage attention maps
        rgb_img, first_stage_attentions = get_single_stage_attention(image_path, model, transform, patch_size, 0, targetHeight, targetWidth, stage_scale, mode = 'bilinear')             
        rgb_img, second_stage_attentions = get_single_stage_attention(image_path, model, transform, patch_size, 1, targetHeight, targetWidth, stage_scale, mode = 'bilinear')
        rgb_img, third_stage_attentions = get_single_stage_attention(image_path, model, transform, patch_size, 2, targetHeight, targetWidth, stage_scale, mode = 'bilinear')
        rgb_img, forth_stage_attentions = get_single_stage_attention(image_path, model, transform, patch_size, 3, targetHeight, targetWidth, stage_scale, mode = 'bilinear')

        # combine the first 3 stages(1+2+5 heads) to show them in 3x3 plot
        combined_attn = [first_stage_attentions, second_stage_attentions, third_stage_attentions]
        combined_attn = np.concatenate(combined_attn, axis=0)

        if show_comb:
            # show first 3 stages
            attentions = combined_attn
            stage = 'first_3_stage'
            j = 0
        else:
            # show last stage
            attentions = forth_stage_attentions
            stage = 'last_stage'
            j = 8

        for i in range(len(axes)):
            axes[i].clear()
            if (i < 4):
                axes[i].imshow(rgb_img)
                axes[i].imshow(attentions[i], cmap='inferno', alpha=0.5)
                axes[i].set_title(titles[i+j], x= 0.20, y=0.9, va="top")
                
            elif(i==4):
                # make middle cell black
                axes[i].imshow(np.zeros_like(rgb_img))
            else:
                axes[i].imshow(rgb_img)
                axes[i].imshow(attentions[i-1], cmap='inferno', alpha=0.5)
                axes[i].set_title(titles[i-1+j], x= 0.20, y=0.9, va="top")

            axes[i].axis('off')

        # save plot
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f'results/attn_imgs/{image_name}_{stage}.png')


def plot_video_and_save_image(video_path, model, transform, patch_size, targetHeight, targetWidth, stage_scale, show_comb):
    # set plot titles
    stage_heads = [1, 2, 5, 8]
    titles = []
    for stage_index, stage_nh in enumerate(stage_heads):
        titles.extend([f"STAGE_{stage_index+1}_HEAD_{x+1}" for x in range(stage_nh)])
    
    # plot the attention images
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 4}
    plt.rc('font', **font)
    plt.rcParams['text.color'] = 'white'
    fig, axes = plt.subplots(3,3, figsize=(19.1,6.1))
    axes = axes.flatten()
    fig.tight_layout()

    counter = 0
    pbar = tqdm(total = 15*78, position=0, leave=True)  # fps=15, 78s
    
    input_handle = cv2.VideoCapture(video_path)
    while input_handle.isOpened():
        ret, frame = input_handle.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # get stage attention maps
            rgb_img, first_stage_attentions = get_single_stage_attention2(frame, model, transform, patch_size, 0, targetHeight, targetWidth, stage_scale, mode = 'bilinear')             
            rgb_img, second_stage_attentions = get_single_stage_attention2(frame, model, transform, patch_size, 1, targetHeight, targetWidth, stage_scale, mode = 'bilinear')
            rgb_img, third_stage_attentions = get_single_stage_attention2(frame, model, transform, patch_size, 2, targetHeight, targetWidth, stage_scale, mode = 'bilinear')
            rgb_img, forth_stage_attentions = get_single_stage_attention2(frame, model, transform, patch_size, 3, targetHeight, targetWidth, stage_scale, mode = 'bilinear')

            # combine the first 3 stages(1+2+5 heads) to show them in 3x3 plot
            combined_attn = [first_stage_attentions, second_stage_attentions, third_stage_attentions]
            combined_attn = np.concatenate(combined_attn, axis=0)

            if show_comb:
                # show first 3 stages
                attentions = combined_attn
                stage = 'first_3_stage'
                j = 0
            else:
                # show last stage
                attentions = forth_stage_attentions
                stage = 'last_stage'
                j = 8

            for i in range(len(axes)):
                axes[i].clear()
                if (i < 4):
                    axes[i].imshow(rgb_img)
                    axes[i].imshow(attentions[i], cmap='inferno', alpha=0.5)
                    axes[i].set_title(titles[i+j], x= 0.20, y=0.9, va="top")
                    
                elif(i==4):
                    # make middle cell black
                    axes[i].imshow(np.zeros_like(rgb_img))
                else:
                    axes[i].imshow(rgb_img)
                    axes[i].imshow(attentions[i-1], cmap='inferno', alpha=0.5)
                    axes[i].set_title(titles[i-1+j], x= 0.20, y=0.9, va="top")

                axes[i].axis('off')
        else:
            break

        # save plot
        fig.subplots_adjust(wspace=0, hspace=0)
        if counter < 10:
            fig.savefig(f'results/attn_imgs/0000{counter}_{stage}.png')
        elif counter < 100:
            fig.savefig(f'results/attn_imgs/000{counter}_{stage}.png')
        elif counter < 1000:
            fig.savefig(f'results/attn_imgs/00{counter}_{stage}.png')
        else:
            fig.savefig(f'results/attn_imgs/0{counter}_{stage}.png')
        counter += 1
        pbar.update(1)

def convert_images_to_video(images_dir, output_video_path, fps : int = 20):
    input_images = [os.path.join(images_dir, *[x]) for x in sorted(os.listdir(images_dir)) if x.endswith('png')]

    if(len(input_images) > 0):
        sample_image = cv2.imread(input_images[0])
        height, width, _ = sample_image.shape
        
        # handles for input output videos
        output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        # create progress bar
        num_frames = int(len(input_images))
        pbar = tqdm(total = num_frames, position=0, leave=True)

        for i in tqdm(range(num_frames), position=0, leave=True):
            frame = cv2.imread(input_images[i])
            output_handle.write(frame)
            pbar.update(1)

        # release the output video handler
        output_handle.release()
    else:
        print("there is no image")


if __name__ == '__main__':
    inputs = 'video'
    NUM_CLASSES = 3    # 3, 19
    patch_size = 8  # 8, 32. Depend on the input image size
    stage_scale = [4, 8, 16, 32]

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model_path = 'trained_model/BDD100K/segformer_mit_b3_bdd_3CLS_CEloss_4Epoch.pt'
    model = segformer_mit_b3(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path))

    # normalize image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # True, shows the first 3 stages; False, shows the last stage
    show_combined_stages = False

    if inputs == 'image':
        # input images
        input_dir = 'dataset/demoVideo/stuttgart_00'
        image_list = sorted(os.listdir(input_dir))
        images_path = [os.path.join(input_dir, x) for x in image_list]
        targetWidth = 1024
        targetHeight = 512
        # plot and save images
        plot_and_save_image(images_path, model, transform, patch_size, targetHeight, targetWidth, stage_scale, show_combined_stages)
    elif inputs == 'video':
        # input video
        video_path = 'dataset/demoVideo/highway_1241_376.avi'
        targetWidth = 1248
        targetHeight = 384
        # plot and save images
        plot_video_and_save_image(video_path, model, transform, patch_size, targetHeight, targetWidth, stage_scale, show_combined_stages)

    # Convert to video
    video_output_dir = os.path.join('.', *['results'])
    if(not os.path.isdir(video_output_dir)):
        os.mkdir(video_output_dir)
    if show_combined_stages:
        output_video_path = os.path.join(video_output_dir, *[f"first_3_stages_demoVideo.mp4"])
    else:
        output_video_path = os.path.join(video_output_dir, *[f"last_stages_demoVideo.mp4"])
    
    convert_images_to_video('results/attn_imgs', output_video_path)