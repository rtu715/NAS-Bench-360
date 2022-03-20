#!/bin/bash

mkdir data 
cd data

# CIFAR-100
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xf cifar-100-python.tar.gz 
rm cifar-100-python.tar.gz 
mkdir cifar-100 
mv cifar-100-python cifar-100 

wget https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz 
gzip -d s2_cifar100.gz 
mkdir spherical 
mv s2_cifar100 spherical

