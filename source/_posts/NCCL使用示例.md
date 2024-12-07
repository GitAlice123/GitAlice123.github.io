---
title: NCCL使用示例
date: 2024-11-29 15:01:43
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---

# 通信组创建和销毁
## 一个进程，一个线程，多个设备
* 在这种单进程的场景下，可以使用ncclCommInitAll()
```c
int main(int argc, char *argv[])
{
    ncclComm_t comms[4];

    // managing 4 devices
    int nDev = 4;
    int size = 32 * 1024 * 1024;
    int devs[4] = {0, 1, 2, 3};

    // 这部分的解释见《NCCL代码中NCCL代码中常用的函数和宏定义》的cudaAlloc部分
    // allocating and initializing device buffers
    float **sendbuff = (float **)malloc(nDev * sizeof(float *));
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    // 单线程控制多个GPU时必须要用group API，否则会死锁
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i], size, ncclFloat, ncclSum,
                                comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);

    printf("Success \n");
    return 0;
}
```