#!/bin/bash

mkdir datasets
cd datasets

# CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xf cifar-100-python.tar.gz 
rm cifar-100-python.tar.gz 
mkdir cifar-100 
mv cifar-100-python cifar-100 

# Spherical
wget https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz 
gzip -d s2_cifar100.gz 
mkdir spherical 
mv s2_cifar100 spherical

# NinaPro
mkdir ninapro
cd ninapro
wget https://pde-xd.s3.amazonaws.com/ninapro/ninapro_train.npy
wget https://pde-xd.s3.amazonaws.com/ninapro/label_train.npy
wget https://pde-xd.s3.amazonaws.com/ninapro/ninapro_val.npy
wget https://pde-xd.s3.amazonaws.com/ninapro/label_val.npy
wget https://pde-xd.s3.amazonaws.com/ninapro/ninapro_test.npy
wget https://pde-xd.s3.amazonaws.com/ninapro/label_test.npy
cd ..

# FSD50K
#wget https://pde-xd.s3.amazonaws.com/audio/audio.zip
#mkdir -p audio/data
#mv audio.zip audio/data
#cd audio/data
#unzip audio.zip
#cd ../..

# Darcy Flow
mkdir darcyflow
cd darcyflow
wget https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat
wget https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat
cd ..

# PSICOV

# Cosmic

# ECG
wget https://pde-xd.s3.amazonaws.com/ECG/challenge2017.pkl
mkdir ecg
mv challenge2017.pkl ecg

# Satellite
mkdir satellite
cd satellite
wget https://pde-xd.s3.amazonaws.com/satellite/satellite_train.npy
wget https://pde-xd.s3.amazonaws.com/satellite/satellite_test.npy
cd ..

# DeepSEA
mkdir deepsea
cd deepsea
wget https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz
cd ..

