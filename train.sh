#!/bin/sh
python main.py fit --trainer.gpus 1 --trainer.max_epochs=-1 --model.lr=0.0003 --data.batch_size=8 --ckpt_path=lightning_logs/version_15/checkpoints/epoch=44-step=34200.ckpt
