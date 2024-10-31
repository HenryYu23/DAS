CUDA_VISIBLE_DEVICES=1 \
python modelSele.py \
    --config ./outputs/AT_c2f/config.yaml \
    --num-gpus 1 \
    --eval-only \
    MODEL.WEIGHTS outputs/AT_c2f/model_0024999.pth