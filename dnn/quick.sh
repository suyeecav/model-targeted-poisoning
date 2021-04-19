#!/bin/bash

python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-3 --poison_rates $3 --epochs 50 --seed 2021
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-2 --poison_rates $3 --epochs 50 --seed 2021
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-1 --poison_rates $3 --epochs 50 --seed 2021
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e0  --poison_rates $3 --epochs 50 --seed 2021

python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-3 --poison_rates $3 --epochs 50 --seed 68
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-2 --poison_rates $3 --epochs 50 --seed 68
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-1 --poison_rates $3 --epochs 50 --seed 68
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e0  --poison_rates $3 --epochs 50 --seed 68

python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-3 --poison_rates $3 --epochs 50 --seed 142
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-2 --poison_rates $3 --epochs 50 --seed 142
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e-1 --poison_rates $3 --epochs 50 --seed 142
python generate_target_dnn.py --study_mode --dataset $1 --verbose --dir_prefix data/nosaveruns/ --poison_class 0 --model_arch $2 --lr 2e-3 --weight_decay 2e0  --poison_rates $3 --epochs 50 --seed 142
