#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

# >>> import os; import torch; print(os.path.dirname(torch.__file__))
# /home/shenwang/.conda/envs/pytorch11/lib/python3.6/site-packages/torch
echo ${TORCH}

#/home/shenwang/.conda/envs/pytorch11/lib/python3.6/site-packages/torch/include/THC

cd src
echo "Compiling channelnorm kernels by nvcc..."
rm ChannelNorm_kernel.o
rm -r ../_ext

# nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52 -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC


# /home/shenwang/.conda/envs/pytorch11/lib/python3.6/site-packages/torch
echo ${TORCH}/include/TH 

# NVCC_OPTS=-O3 -arch=sm_50 -Xcompiler -Wall -Xcompiler -Wextra -m64
# NVCC_OPTS=-x cu -Xcompiler -fPIC -arch=sm_70

# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation

nvcc -c -o ChannelNorm_kernel.o ChannelNorm_kernel.cu -x cu -arch=sm_75 -I ${TORCH}/include/TH -I ${TORCH}/include/THC -I ${TORCH}/include

# cd ../
# python build.py
# echo "all done"