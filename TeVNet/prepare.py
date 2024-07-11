import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import os

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    hr_patches = []
    n = 0
    for image_path in tqdm(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        hr = np.transpose(hr, (2,0,1))
        # print(hr.shape)
        hr_patches.append(hr)

        # n = n+1
        # if n == 100:
        #     break
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=hr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()
    


def eval(args):
    
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    # print(sorted(glob.glob('{}/*'.format(args.images_dir))))
    for i, image_path in tqdm(enumerate(sorted(glob.glob('{}/*'.format(args.images_dir))))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        hr = np.transpose(hr, (2,0,1))
        
        lr_group.create_dataset(str(i), data=hr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    
    # os.makedirs(args.output_path, exist_ok=True)

    if not args.eval:
        train(args)
    else:
        eval(args)
