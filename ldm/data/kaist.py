import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as tf

def my_transform(image1, image2):
    # # 50%的概率应用垂直，水平翻转。
    # if random.random() > 0.5:
    #     image1 = tf.hflip(image1)
    #     image2 = tf.hflip(image2)

    return image1, image2

class KAISTBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 sam=False,
                 sam_root=None
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        self.sam_root = sam_root
        self.sam = sam
        
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
                    # "sam_file_path_": [os.path.join(self.sam_root, l)
                        #    for l in self.image_paths] if self.sam else None,

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # seed
        seed = np.random.randint(2147483647)
        
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
        if self.size is not None:
            image_ir = image_ir.resize((self.size, self.size), resample=self.interpolation)

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
            image_vi = image_vi.resize((self.size, self.size), resample=self.interpolation)
        
        # set same seed to aug!!
        # random.seed(seed)
        # image_vi = self.flip(image_vi)
        
        
        if self.sam:
            ## preprocesing sam imgs
            image_sam = Image.open(example["sam_file_path_"])
            if not image_sam.mode == "RGB":
                image_sam = image_sam.convert("RGB")

            # default to score-sde preprocessing
            img_sam = np.array(image_sam).astype(np.uint8)
            crop = min(img_sam.shape[0], img_sam.shape[1])
            h, w, = img_sam.shape[0], img_sam.shape[1]
            img_sam = img_sam[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

            image_sam = Image.fromarray(img_sam)
            if self.size is not None:
                image_sam = image_sam.resize((self.size, self.size), resample=self.interpolation)
            
            # set same seed to aug!!
            # random.seed(seed)
            # image_sam = self.flip(image_sam)
            image_sam = np.array(image_sam).astype(np.uint8)
            image_cond = np.concatenate((image_vi, image_sam), axis = -1)
        else:
            image_cond = image_vi
        
        image_ir, image_vi = my_transform(image_ir, image_vi)
        image_ir = np.array(image_ir).astype(np.uint8)
        image_vi = np.array(image_vi).astype(np.uint8)
        example["image"] = (image_ir / 127.5 - 1.0).astype(np.float64)
        example["conditional"] = (image_vi / 127.5 - 1.0).astype(np.float64)
        
        return example


class KAISTTrain(KAISTBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/kaist/KAIST_00_01_03_04.txt", 
                         data_root="/public/home/maofangyuan/dataset/KAIST_00_01_03_04", 
                         **kwargs)


class KAISTVal(KAISTBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/kaist/KAIST_02_05.txt", 
                         data_root="/public/home/maofangyuan/dataset/KAIST_02_05",
                         flip_p=flip_p, **kwargs)

class KAISTTrain_SAM(KAISTBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/kaist/KAIST_00_01_03_04.txt", 
                         data_root="/home/maofangyuan/dataset/KAIST_00_01_03_04", 
                         sam=True, 
                         sam_root="/home/maofangyuan/dataset/SAMresults/KAIST_00_01_03_04/visible/rgb",
                         **kwargs)


class KAISTVal_SAM(KAISTBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/kaist/KAIST_02_05.txt",
                         data_root="/home/maofangyuan/dataset/KAIST_02_05",
                         flip_p=flip_p, 
                         sam=True, 
                         sam_root="/home/maofangyuan/dataset/SAMresults/KAIST_02_05/visible/rgb",
                         **kwargs)



# class LSUNBedroomsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


# class LSUNBedroomsValidation(LSUNBase):
#     def __init__(self, flip_p=0.0, **kwargs):
#         super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
#                          flip_p=flip_p, **kwargs)


# class LSUNCatsTrain(LSUNBase):
#     def __init__(self, **kwargs):
#         super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


# class LSUNCatsValidation(LSUNBase):
#     def __init__(self, flip_p=0., **kwargs):
#         super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
#                          flip_p=flip_p, **kwargs)
