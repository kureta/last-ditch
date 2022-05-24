#!/bin/sh
tensorboard --logdir /workspace/container_1/lightning_logs --bind_all --port 80 &
python main.py fit --trainer.gpus 1 --trainer.max_epochs=-1 --trainer.default_root_dir=/workspace/container_1/ --model.lr=0.0003 --data.batch_size=16 --data.path=/workspace/container_1/features.pth
