CUDA_VISIBLE_DEVICES=0 python scripts/rgb2ir_vqf8.py --steps 200 \
--indir /path/to/image \
--outdir /path/to/result \
--config /path/to/config \
--checkpoint /path/to/checkpoint \
--ddim_eta 0.0