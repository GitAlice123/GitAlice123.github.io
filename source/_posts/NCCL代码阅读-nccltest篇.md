---
title: NCCL代码阅读-nccltest篇
date: 2024-12-28 14:56:55
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---
## 本篇内容简介
* 我们以AllReduce操作为例，来说明整个nccl-test的调用栈
* 我们的场景是单机，两卡，两个进程（一个进程管一个卡）
* 我们输入的shell命令主要如下：
```shell
mpirun --allow-run-as-root -n 2 ./build/all_reduce_perf -f 2 -b 1m -e 1m -C 1
mpirun --allow-run-as-root -n 2 ./build/all_reduce_perf -f 2 -b 1m -e 1m
```
* 解释一下各参数
    * -f是factor，数据大小每次增长倍数
    * -b是begin，-b 1m表示数据大小从1m开始增长
    * -e是end，-e 1m表示数据大小增长到1m
    * -C是cpu时间，-C为1表示最终数据会打印CPU的时间（同步部分），不含下面GPU异步执行的时间，CPU做完他的工作就返回了；反之就显示等待GPU完成工作的总时间
## main
* common.cu中的main函数，是整个程序的入口
* 根据命令parse之后，写了这么几个参数
    * nThread（我们这个命令下就是1）
    * nGpus（我们这个命令下就是1）
    * minBytes（就是-b的值换成B为单位）
    * maxBytes（就是-e的值换成B为单位）
    * step（-f）
    * warmup iters（默认是5）
    * iters（默认是20）
    * agg iters（默认是1）
    * validation（默认是1） 
    * graph（默认是0）
    * 还有一些其他的参数，我们的命令里面都没加，有需要的去README里面对着看看
* 然后进入run函数

## run
### 初始化MPI环境和NCCL配置
* 会让每个MPI进程获取到自己的rank
* 还会按照color对进程分组（这个我们的命令用不到）
### 设备初始化和内存分配
* 打印设备信息
* 计算每个进程可以使用的最大内存，由设备当前内存来确定
### NCCL初始化
* **生成NCCL唯一的ID**，由第一个进程生成这个ID，然后用MPI广播给所有进程
* 这边另外提一下，每个MPI进程里面的comm数量=nthreads*nGPUs（每个进程要管的GPU数量），nGPUs不是物理上有多少个GPU，而是一个线程要管一个GPU的话，就要有一个comm来作为控制块
* 每个进程分配的sendBuffer和recvBuffer的数量也是nthreads*nGPUs
* 根据 ncclProc（当前进程在 NCCL 子集中的 rank）来初始化 NCCL 通信。对于多个进程，需要通过 ncclCommInitRank 初始化每个进程的 NCCL 通信，主要就是初始化comm结构体
### 内存分配
* 为每个 GPU 分配内存，用于存储发送和接收数据的缓冲区（sendbuffs 和 recvbuffs）
* expected部分用来验证传输的数据结果是否正确，开启dataCheck的时候有用
### 设置线程的运行参数
* 要设置的就是这些线程：testThread threads[nThreads]
* 设置threads[t].args的值，这个args会被作为所有任务的信息传到nccl中执行
### 进入threadRunTests 
## threadRunTests
* cudaSetDevice指定一下后面要用的GPU是哪张
* 进入ncclTestEngine.runTest，这里allreduce会进入allReduceEngine的runTest
    * 是怎么确定要进入allReduceEngine的请见《pragma-weak初遇》
## AllReduceRunTest
* 主要就是进入TimeTest，传的最重要的参数还是刚刚上面的args
* 顺便提一下，nccltype（后面的type等，总之是数据类型）默认是ncclFloat（查一下表就是float），ncclop默认是ncclSum
## TimeTest
* 跑几个warmup的iter
* 进入BenchTime，跑两个实验，一个是in-place，一个是相反的
## BenchTime
* 首先会算一个count，一共有多少个**数据元素**
```c
size_t count = args->nbytes / wordSize(type);
```
* 然后开始startColl，这就是真正要下去执行的地方了
* 注意下面有两个时间
```c
  double cputimeSec = tim.elapsed()/(iters*agg_iters);
  TESTCHECK(completeColl(args));

  double deltaSec = tim.elapsed();
  deltaSec = deltaSec/(iters*agg_iters);
  if (cudaGraphLaunches >= 1) deltaSec = deltaSec/cudaGraphLaunches;
  Allreduce(args, &deltaSec, average);
```
* cputimeSec测的是在CPU把任务下发之后就返回的时间，而deltaSec测的是等到completeColl结束之后的总时间，要等待GPU完成

## startColl
* 上面会对in-place的方法给出一些内存的偏移，防止重叠
* 主要函数就是这个：
```c
    TESTCHECK(args->collTest->runColl(
          (void*)(in_place ? recvBuff + args->sendInplaceOffset*rank : sendBuff),
          (void*)(in_place ? recvBuff + args->recvInplaceOffset*rank : recvBuff),
        count, type, op, root, args->comms[i], args->streams[i]));
```
## runColl
* 相同的父类实例化方法，往下找会找到，在AllReduce中我们实际调用的是AllReduceRunColl
* 接下来调用
```c
NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
```
* 正式进入NCCL的逻辑！