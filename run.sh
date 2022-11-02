#!/bin/bash
# python ./attack/badnet_attack.py --yaml_path ../config/attack/badnet/cifar10.yaml --dataset cifar10 --dataset_path ../data --save_folder_name badnet_0_1
# python attack/badnet_attack.py --yaml_path ../config/attack/badnet/sbi_0.5x0.5.yaml --dataset sbi --dataset_path ../data --save_folder_name sbi_0.5x0.5
python attack/badnet_attack.py --yaml_path ../config/attack/badnet/sbi_9by9_380by380.yaml --dataset sbi --dataset_path ../data --save_folder_name sbi_9by9_380by380 