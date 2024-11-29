---
title: NCCL中重要的数据结构
date: 2024-11-29 10:06:36
tags:
---

# struct ncclComm
实际使用的时候是
```c
typedef struct ncclComm* ncclComm_t;
```
* 在src\nccl.h.in
* 通信上下文
开会，那么这个通信上下文就是会议室

# struct ncclInfo

