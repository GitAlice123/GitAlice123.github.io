---
title: NCCL代码阅读-01
date: 2024-11-28 15:45:03
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---
# 创建一个通信组(communicator)                                                 
* 创建一个通信组之前，每个CUDA设备都要被分配一个唯一的rank id
* 有了这个rank id和CUDA设备的静态映射，ncclCommInitRank(), ncclCommInitRankConfig() and ncclCommInitAll() 三个函数会创建communicator objects，每个communicator object会和一个固定的rank（及一个CUDA设备）关联。
* 在调用ncclCommInitRank之前，需要调用ncclGetUniqueId()来获取一个unique id，这个ID必须广播到所有参与通信的进程，让他们知道自己在communicator中
* 比如有四个GPU互相通信，加入了一个通信组，那么这个通信组就需要一个通信上下文记录所有的信息
* 类比四个人开会，那么这个通信上下文就是会议室

## ncclCommInitRank
```c
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
```
* 创建一个communicator object
* 里面调用ncclCommInitRankDev()

## ncclCommInitAll
* 在**一个CPU进程**里面执行(**因此他后面所调用的所有函数都是在这一个进程，一个线程里面执行的**)，创建多个communicator object
* 但是只能是单进程版本，也因此不支持多node通信
* 首先检查了各种数据的有效性
* 然后调用ncclGetUniqueId()获取一个unique id
    * ncclGetUniqueId()首先调用ncclInit()初始化NCCL

## ncclInit()
* 这是一个在所有线程中只会执行一次的函数
* 在两个地方被调用：ncclGetUniqueId和ncclCommInitRankDev
* 如果是ncclGetUniqueId调用的，那么分两种情况：
    * 在ncclCommInitAll中调用，那其实就一个进程，一个线程，不用担心会被多次调用
    * 在ncclCommInitRank前面调用，那么就要限制只有第一个线程调用，后面的线程不会再调用


