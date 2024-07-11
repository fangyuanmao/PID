import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as tf
from copy import deepcopy

def random_crop(image1, image2):
    min_ratio = 0.5
    max_ratio = 1

    w, h = image1.size
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image1 = image1.crop((x, y, x + new_w, y + new_h))
    image2 = image2.crop((x, y, x + new_w, y + new_h))

    return image1, image2


class my_transform_crop():
    def __init__(self, crop_p=0.5):
        self.crop_p=crop_p
    def crop_enhance(self, image1, image2):
        if random.random() <= self.crop_p:
            image1, image2 = random_crop(image1, image2)
        return image1, image2

class my_transform_flip():
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p
        
    def flip_enhance(self, image1, image2):
        if random.random() <= self.flip_p:
            image1 = np.flip(image1, axis=1)
            image2 = np.flip(image2, axis=1)
        return image1, image2

class my_transform_gray():
    def __init__(self, gray_p=0.5):
        self.gray_p = gray_p
        
    def gray_enhance(self, image1):
        if random.random() <= self.gray_p:
            gray = deepcopy(0.299*image1[:,:,0] + 0.587*image1[:,:,1] + 0.114*image1[:,:,2])
            image1[:,:,0] = gray
            image1[:,:,1] = gray
            image1[:,:,2] = gray
        return image1

class KAISTBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        
        self.ir_data_root = os.path.join(self.data_root, 'lwir')
        self.vi_data_root = os.path.join(self.data_root, 'visible')
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "ir_file_path_": [os.path.join(self.ir_data_root, l)
                           for l in self.image_paths],
            "vi_file_path_": [os.path.join(self.vi_data_root, l)
                           for l in self.image_paths],
        }
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip_enhance = my_transform_flip(flip_p=flip_p)
        self.gray_enhance = my_transform_gray(gray_p=flip_p)
        self.crop_enhance = my_transform_crop(crop_p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # return batch
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        ## preprocessing ir imgs
        image_ir = Image.open(example["ir_file_path_"])
        if not image_ir.mode == "RGB":
            image_ir = image_ir.convert("RGB")

        # default to score-sde preprocessing
        img_ir = np.array(image_ir).astype(np.uint8)
        crop = min(img_ir.shape[0], img_ir.shape[1])
        h, w, = img_ir.shape[0], img_ir.shape[1]
        img_ir = img_ir[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image_ir = Image.fromarray(img_ir)

        ## preprocesing visible imgs
        image_vi = Image.open(example["vi_file_path_"])
        if not image_vi.mode == "RGB":
            image_vi = image_vi.convert("RGB")

        # default to score-sde preprocessing
        img_vi = np.array(image_vi).astype(np.uint8)
        crop = min(img_vi.shape[0], img_vi.shape[1])
        h, w, = img_vi.shape[0], img_vi.shape[1]
        img_vi = img_vi[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image_vi = Image.fromarray(img_vi)
        
        if self.size is not None:
            image_ir, image_vi = self.crop_enhance.crop_enhance(image_ir, image_vi)
            image_ir = image_ir.resize((self.size, self.size), resample=self.interpolation)
            image_vi = image_vi.resize((self.size, self.size), resample=self.interpolation)
        
        image_ir = np.array(image_ir).astype(np.uint8)
        image_vi = np.array(image_vi).astype(np.uint8)
        
        # image_vi = self.gray_enhance.gray_enhance(image_vi)
        image_ir, image_vi = self.flip_enhance.flip_enhance(image_ir, image_vi)
        
        example["image"] = (image_ir / 127.5 - 1.0).astype(np.float64)
        example["conditional"] = (image_vi / 127.5 - 1.0).astype(np.float64)
        return example


class KAISTTrain(KAISTBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/KAIST512/KAIST_512_train.txt", 
                         data_root="/public/home/maofangyuan/dataset/KAIST_512/train", 
                         **kwargs)


class KAISTVal(KAISTBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/KAIST512/KAIST_512_test.txt", 
                         data_root="/public/home/maofangyuan/dataset/KAIST_512/test",
                         flip_p=flip_p, **kwargs)