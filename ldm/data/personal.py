import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Personalize0(Dataset):
    def __init__(self,
                 txt_file,
                 size=128,
                 interpolation="bicubic",
                 flip_p=0.5,
                 is_mri=True,
                 val=False
                 ):
        self.data_paths = txt_file
        self.is_mri=is_mri
        self.image_paths=pd.read_csv(self.data_paths)["Image_path"]
        if is_mri:
            self.mri_paths=pd.read_csv(self.data_paths)["MRI_path"]
        if val:
            self.mri_paths=self.mri_paths[:1024]
            self.image_paths=self.image_paths[:1024]
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [l for l in self.image_paths],
        }
        self._length = len(self.image_paths)
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
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(self.image_paths[i])
        if self.is_mri:
            mri=nib.load(self.mri_paths[i]).get_fdata()/2000
            x,y,z,t=mri.shape
            MRI=np.zeros((5,72,88,72))
            MRI[:,:min(x,72),:min(y,88),:min(z,72)]=mri[:min(x,72),:min(y,88),:min(z,72),:].transpose(3,0,1,2)
            #MRI = zoom(MRI, (1,0.75, 0.75, 0.75))
            example["mri"]=MRI.astype(np.float32)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class PersonalizeTrain0(Personalize0):
    def __init__(self, **kwargs):
        super().__init__(txt_file=csv_path_train)

class PersonalizeVal0(Personalize0):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file=csv_path_val ,val=True,
                         flip_p=flip_p)