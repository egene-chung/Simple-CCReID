# The code is builded with DistributedDataParallel. 
# Reprodecing the results in the paper should train the model on 2 GPUs.
# You can also train this model on single GPU and double config.DATA.TRAIN_BATCH in configs.
# LTCC dataset
# 2개의 GPU 사용할 땐 nproc per mode를 2로 설정하고, 1개 사용할 땐 1로 설정
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 4
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py --dataset ltcc --cfg configs/expB_dyn_eps.yaml --gpu 4
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py --dataset ltcc --cfg configs/expC_temp_sweep.yaml --gpu 4
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py --dataset ltcc --cfg configs/expD_large_batch.yaml --gpu 4 
# python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py --dataset ltcc --cfg configs/expE_momentum_alpha.yaml --gpu 4
# # For PRCC dataset
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 #
# # For VC-Clothes dataset. You should change the root path of '--resume' to your output path.
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes --cfg configs/res50_cels_cal.yaml --gpu 0,1 #
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes_cc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --eval --resume /data/guxinqian/logs/vcclothes/res50-cels-cal/best_model.pth.tar #
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset vcclothes_sc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --eval --resume /data/guxinqian/logs/vcclothes/res50-cels-cal/best_model.pth.tar #
# # For DeepChange dataset. Using amp can accelerate training.
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset deepchange --cfg configs/res50_cels_cal_16x4.yaml --amp --gpu 0,1 #
# # For LaST dataset. Using amp can accelerate training.
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset last --cfg configs/res50_cels_cal_tri_16x4.yaml --amp --gpu 0,1 #
# # For CCVID dataset
# python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ccvid --cfg configs/c2dres50_ce_cal.yaml --gpu 0,1 #
#!/bin/bash