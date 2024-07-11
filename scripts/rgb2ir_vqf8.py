import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import sys
sys.path.append(r'.')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, device):
    image = np.array(Image.open(image).convert("RGB")).astype(np.uint8)
    
    crop = min(image.shape[0], image.shape[1])
    h, w, = image.shape[0], image.shape[1]
    image = image[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
    image = Image.fromarray(image)
    image = image.resize((512, 512), resample=Image.BICUBIC)
    
    image = np.array(image)
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    
    image = torch.from_numpy(image)

    batch = {"conditional": image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0
    )
    opt = parser.parse_args()

    visibles = sorted(glob.glob(os.path.join(opt.indir, "*.jpg")))
    if len(visibles) == 0:
        visibles = sorted(glob.glob(os.path.join(opt.indir, "*.png")))
    print(f"Found {len(visibles)} inputs.")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.checkpoint,map_location='cuda:0')["state_dict"],
                          strict=False)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            i = 0
            for visible in tqdm(visibles):
                outpath = os.path.join(opt.outdir, os.path.split(visible)[1])
                batch = make_batch(visible, device=device)

                c = model.cond_stage_model.encode(batch["conditional"])
                shape = (c.shape[1]+1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False,
                                                 ddim_eta=opt.ddim_eta)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)
                
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(predicted_image.astype(np.uint8)).save(outpath)