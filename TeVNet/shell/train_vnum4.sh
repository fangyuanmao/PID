CUDA_VISIBLE_DEVICES=0 \
python train.py \
--smp_model Unet --smp_encoder resnet18 --smp_encoder_weights imagenet \
--num-epochs 1000 \
--num-epochs-save 50 \
--num-epochs-val 10 \
--outputs-dir /path/to/save \
--batch-size 64 \
--lr 0.001 \
--train-dir /path/to/train/data \
--eval-dir /path/to/val/data \
--vnums 4