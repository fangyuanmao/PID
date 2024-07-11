import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
transform = T.ToPILImage()
from PIL import Image, ImageDraw

import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import os
import random


import torch
from torch.utils.data import Dataset, Subset

def draw_rectangle_over_mask(image):
    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # Find white pixel coordinates
    white_pixels = np.where(img_array > 1)

    # Calculate bounding box coordinates with 3-pixel margins
    min_x, min_y = np.min(white_pixels[1]) - 3, np.min(white_pixels[0]) - 3
    max_x, max_y = np.max(white_pixels[1]) + 3, np.max(white_pixels[0]) + 3

    # Ensure the coordinates are within the image bounds
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, img_array.shape[1])
    max_y = min(max_y, img_array.shape[0])

    # Draw rectangle on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle([(min_x, min_y), (max_x, max_y)], fill='white')

    return image



class InpaintingTrain_autoencoder(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.data_root=data_root
        self.images = os.listdir(data_root)


    def generate_stroke_mask(self, im_size, parts=15, maxVertex=25, maxLength=80, maxBrushWidth=60, maxAngle=360):
        
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        for i in range(parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        return mask

    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):

        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        
        return mask



    def __len__(self):
        return len(self.images)


    def __getitem__(self, i):
        
        image = np.array(Image.open(self.data_root+"/"+self.images[i]).convert("RGB").resize((512,512)))
        image = image.astype(np.float32) / 255.0#
        # image = image[None].transpose(0,3,1,2)
        image = torch.from_numpy(image)

        
        mask = self.generate_stroke_mask([self.size, self.size])
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        # mask = mask[None].transpose(0,3,1,2)

        
        
        mask = torch.from_numpy(mask)
        masked_image = (1 - mask) * image
        
        

        ##50% chance to return a masked_image instead of the original image.
        if random.uniform(0, 1)<0.5:
            batch = {"image": np.squeeze(image,0), "masked_image": np.squeeze(masked_image,0)}
        else:
            batch = {"masked_image": np.squeeze(image,0), "image": np.squeeze(masked_image,0)}
        
        batch = {"image": np.squeeze(image,0), "masked_image": np.squeeze(masked_image,0)}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch
        



class InpaintingTrain_ldm(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.data_root=data_root
        self.mask_list = []
        
        for mask in os.listdir(data_root+"/masks"):
            self.mask_list.append(mask)

                    
    def __len__(self):
        return len(self.mask_list)


    def __getitem__(self, i):
        
        mask_address=self.mask_list[i]
        
        mask = Image.open(self.data_root+"/masks/"+mask_address).convert("L").resize((512,512))
        mask = np.array(draw_rectangle_over_mask(mask))
        mask = np.expand_dims(mask, axis=2)
        
        
        
        image = np.array(Image.open(self.data_root+"/images/"+mask_address).convert("RGB").resize((512,512)))
        image = image.astype(np.float32) / 255.0#
        #image = image[None].transpose(0,3,1,2)
        image = torch.from_numpy(image)
        
        mask = mask.astype(np.float32) / 255.0#
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        #mask = mask[None].transpose(0,3,1,2)
        
        
        mask = torch.from_numpy(mask)
        masked_image = (1 - mask) * image
        

        batch = {"image": np.squeeze(image,0), "mask": np.squeeze(mask,0), "masked_image": np.squeeze(masked_image,0)}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch