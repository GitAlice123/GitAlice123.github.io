---
title: NCCL使用示例
date: 2024-11-29 15:01:43
tags:
---

# 通信组创建和销毁
## 一个进程，一个线程，多个设备
* 在这种单进程的场景下，可以使用ncclCommInitAll()
* 下面的例子创建了一个有四个设备的通信组
```c
ncclComm_t comms[4];
int devs[4] = { 0, 1, 2, 3 };
ncclCommInitAll(comms, 4, devs);
```