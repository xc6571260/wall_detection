"""
This code was coding at 2024/8/13 **Guang-Jyun, Jiang
"""
"""
This function is Use YOLOv8 to predict wall edge. Mask is blue (0,0,255)
"""
from ultralytics import YOLO
import os
import torch
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt


def generate_mask(ori_path,model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = Image.open(ori_path)

    width, height = img.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32

    # Crop image to a multiple of 32 for processing
    crop_img = img.crop((0, 0, new_width, new_height))
    img = crop_img


    # Split into blocks of 1024x1024
    block_size = 1024
    width, height = img.size
    blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if (x + block_size) >= width:
                right_edge = width
            else:
                right_edge = x + block_size
            if (y + block_size) >= height:
                down_edge = height
            else:
                down_edge = y + block_size

            block = img.crop((x, y, right_edge, down_edge))
            blocks.append(block)

    mask_list = []
    for block_img in blocks:
        width_small, height_small = block_img.size

        # Convert the image to the format acceptable by the model (e.g., PyTorch Tensor)
        block_img_tensor = torch.tensor(np.array(block_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Make predictions using the model
        results = model.predict(block_img_tensor.to(device))
        background = torch.zeros(height_small, width_small, 3, device=device)  # White background
        if results[0].masks is None:
            mask_list.append(background.cpu().numpy())
        else:
            mask = results[0].masks.data
            box = results[0].boxes

            # Process masks
            for i in range(mask.shape[0]):
                if box[i].cls.item() == 2.0:  # 检查目标类别是否为1
                    # 标记为蓝色区域
                    mask_color = torch.where(mask[i].unsqueeze(2) == 1,
                                             torch.tensor([0, 0, 1], device=device).float(),
                                             torch.tensor([0, 0, 0], device=device).float())
                    mask_color = mask_color * 255
                    background += mask_color

            mask_list.append(background.cpu().numpy().astype(np.uint8)) 

    # Reassemble masks into a mask of the same size as the original image
    reconstructed_mask = np.ones((height, width, 3))  # Initialize with white background
    index = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_mask = mask_list[index]
            block_height, block_width, _ = block_mask.shape
            # Calculate the offset of block_mask in reconstructed_mask
            y_offset = y
            x_offset = x
            # Adjust offsets if block_mask exceeds the range of reconstructed_mask
            if y + block_height > height:
                y_offset -= (y + block_height - height)
            if x + block_width > width:
                x_offset -= (x + block_width - width)
            # Place block_mask in the correct position
            reconstructed_mask[y_offset:y_offset + block_height, x_offset:x_offset + block_width, :] = block_mask[
                                                                                                       :block_height,
                                                                                                       :block_width,
                                                                                                       :]
            index += 1

    # Save the resulting mask
    return reconstructed_mask


