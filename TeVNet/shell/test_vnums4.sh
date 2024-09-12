CUDA_VISIBLE_DEVICES=0 \
python test.py \
--weights-file /path/to/checkpoint \
--image-dir /path/to/data \
--smp_model Unet --smp_encoder resnet34 \
--output-dir /path/to/output \
--vnums 4