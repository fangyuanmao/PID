
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --base configs/latent-diffusion/kaist512.yaml -t --gpus 0,1,2,3,4,5,6,7 \
--resume /path/to/checkpoint