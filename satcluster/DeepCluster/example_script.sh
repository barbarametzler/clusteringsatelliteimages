#!/bin/bash

#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=32:mem=250gb:ngpus=1:gpu_type=RTX6000

DIR="/rds/general/user/abm1818/projects/ssa_satellite_imagery/live/Raster_data/A_Da_De_Ki/"
ARCH="vgg16" #or alexnet/vgg16
LR=0.01
WD=-5
CLUSTERING=Kmeans #Kmeans
K=16 #10000
WORKERS=4
EXP="/rds/general/user/abm1818/projects/ssa_satellite_imagery/live/DeepCluster/Accra_DES_experiments/ADDK_k16_imgnet_lr001_sf/"
BATCH=128
RESUME="/rds/general/user/abm1818/projects/ssa_satellite_imagery/live/DeepCluster/Accra_DES_experiments/ADDK_k16_imgnet_lr001_sf/checkpoint.pth.tar"
#PYTHON="/private/home/${USER}/test/conda/bin/python"
#RESUME="/rds/general/user/abm1818/home/GitHub/deepcluster-master/exp/maxar_256_vgg16_k8_w4_noshuff_norm_lr05_nobwimgs/checkpoints/checkpoint_0.0.pth.tar"
FEPOCH=2
FNAME="/rds/general/user/abm1818/projects/ssa_satellite_imagery/live/DeepCluster/Accra_DES_experiments/ADDK_k16_imgnet_lr001_epoch20_"
mkdir -p ${EXP}

module load anaconda3/personal

#CUDA_VISIBLE_DEVICES=0 
python3 /rds/general/user/abm1818/home/GitHub/deepcluster-master/main_sfeatures.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --clustering ${CLUSTERING} --verbose --batch ${BATCH} --workers ${WORKERS} --resume ${RESUME} --features_name ${FNAME} --features_epoch ${FEPOCH}
