---
title: pragma weak初遇
date: 2024-12-22 22:49:46
tags:
- pragma-weak
- 代码阅读
categories:
- 语言
---
## 第一次见到pragma weak的位置
* 我在看nccl-test代码的时候，在src下面有很多集合操作的源代码，项目结构是这样的：
```shell
.
├── build
│   ├── all_gather_perf
│   ├── all_reduce_perf
│   ├── alltoall_perf
│   ├── broadcast_perf
│   ├── gather_perf
│   ├── hypercube_perf
│   ├── reduce_perf
│   ├── reduce_scatter_perf
│   ├── scatter_perf
│   ├── sendrecv_perf
│   ├── timer.o
│   └── verifiable
│       └── verifiable.o
├── doc
│   └── PERFORMANCE.md
├── gotAcc.py
├── LICENSE.txt
├── Makefile
├── README.md
├── src
│   ├── all_gather.cu
│   ├── all_reduce.cu
│   ├── alltoall.cu
│   ├── broadcast.cu
│   ├── common.cu
│   ├── common.h
│   ├── gather.cu
│   ├── hypercube.cu
│   ├── Makefile
│   ├── nccl1_compat.h
│   ├── reduce.cu
│   ├── reduce_scatter.cu
│   ├── scatter.cu
│   ├── sendrecv.cu
│   ├── timer.cc
│   └── timer.h
└── verifiable
    ├── inexact_regress.cu
    ├── Makefile
    ├── verifiable.cu
    ├── verifiable.h
    └── verifiable.mk
```
* all_gather.cu, all_reduce.cu等等那一大堆文件里面，最后都会有这样一句话：
```c
#pragma weak ncclTestEngine=/*操作名*/Engine
// 下面是all_reduce.cu里的例子
#pragma weak ncclTestEngine=allReduceEngine
```
* 搜了一下，pragma weak是这样的意思，以下摘自GPT

### GPT的答案
`#pragma weak` 是一个预处理指令，用于声明一个符号的弱链接（weak linkage）。这意味着，如果在链接时没有找到该符号的定义，链接器将使用该弱符号声明所提供的定义作为默认值。

#### 这行代码的作用：
1. **声明 `ncclTestEngine` 为弱符号（Weak Symbol）**：
   - 该行代码告诉编译器，`ncclTestEngine` 是一个弱符号，并且它的默认值是 `sendRecvEngine`。这意味着，链接器会在链接时优先使用 `sendRecvEngine` 作为 `ncclTestEngine` 的定义，如果链接器找不到 `ncclTestEngine` 的其他定义。
   
2. **符号的替代：**
   - `ncclTestEngine` 在其他地方可能会被定义为一个不同的符号，通常它是指向某个具体测试引擎实现的函数指针结构体。通过 `#pragma weak`，你可以在某些情况下使用 `sendRecvEngine` 来代替 `ncclTestEngine`，特别是在没有明确提供 `ncclTestEngine` 定义的情况下。
   
3. **动态选择测试引擎：**
   - 如果其他地方（可能是链接时的设置）提供了对 `ncclTestEngine` 的定义，那么编译器会优先使用其他定义。否则，`sendRecvEngine` 将成为 `ncclTestEngine` 的默认值，提供一组测试引擎实现。这样就可以根据实际情况灵活选择不同的引擎实现，而不需要修改代码中的其他部分。

#### 具体场景：
1. **默认引擎实现：**
   `sendRecvEngine` 作为默认的测试引擎包含了四个主要的函数指针：`SendRecvGetBuffSize`、`SendRecvRunTest` 等。当 `ncclTestEngine` 没有被其他地方定义时，`sendRecvEngine` 将会作为默认的实现使用。

2. **符号覆盖：**
   如果在程序的其他地方（可能是其他测试模块或者测试配置中）定义了 `ncclTestEngine`，那么编译器将使用该定义。如果没有定义 `ncclTestEngine`，那么 `sendRecvEngine` 将自动成为 `ncclTestEngine` 的实现。

#### 为什么使用 `#pragma weak`：
`#pragma weak` 通常用于以下场景：
- **库的可扩展性**：允许不同的模块或库扩展并替换默认的符号定义，而不影响现有的代码结构。这种方式允许根据需要替换实现，例如切换到不同的测试引擎或不同的算法实现。
- **避免链接错误**：如果某些符号没有明确的定义，而你又不想因为没有定义而导致链接失败，使用 `#pragma weak` 可以提供一个默认实现。

-----
### 在nccl-test中的使用
* OK，现在我们知道了，pragma weak会在链接的时候给出一个实现，那这么多个弱链接，到底用哪个呢？
* 以sendrecv.cu为例，我们如果想要跑sendrecv，那么我们会输入类似这样的命令：
```shell
./build/sendrecv_perf -f 2 -b 1k -e 1k
```
* 那这个sendrecv_perf就是可执行文件，他的链接用到了哪些文件呢？通过查看makefile，我们可以得知使用了
    * sendrecv.o
    * common.o
    * timer.o
    * 来自 verifiable.mk 的目标文件（例如 verifiable.o）
* 那可以看到了，链接的时候就用到了sendrecv.o这一个，而通篇都没有对ncclTestEngine的强定义，那么就会使用sendrecv中给的弱链接定义啦