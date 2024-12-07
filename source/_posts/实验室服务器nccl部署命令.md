---
title: 实验室服务器nccl部署命令
date: 2024-12-04 16:53:34
tags:
- nccl
- 实验室
categories:
- 实验室实践
---

# nccl编译
```bash
make src.build CUDA_HOME=/usr/lib/nvidia-cuda-toolkit/ NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

# 查看各种库的安装路径
* 由于管理员最初好像是用apt安装的，所以可以这样查找
* 以查找mpi为例
```bash
dpkg -S mpicc
```

# nccl-test编译
```bash
make MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7a1 CUDA_HOME=/usr/lib/nvidia-cuda-toolkit/ NCCL_HOME=/home/cyu/tccl-2024/nccl/build
```

# nccl-test测试
```bash
export LD_LIBRARY_PATH=/home/cyu/tccl-2024/nccl/build/lib:$LD_LIBRARY_PATH
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
```