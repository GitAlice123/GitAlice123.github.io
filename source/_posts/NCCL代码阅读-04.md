---
title: NCCL代码阅读-04
date: 2024-12-15 10:39:02
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---

好的，再更具体一步，假设我们在 **两个 GPU 做 AllReduce** 时，数据的具体划分、传输，以及每一阶段的变化都会明确说明。

---

### **场景再具体化**

假设：
- **数据大小：16 个 `float` 元素**（64 字节，总数据量很小，便于解释）。
- **数据类型：`float`**，每个元素 4 字节。
- **2 个 GPU**（`nranks = 2`），使用环形拓扑（Ring）。
- **1 个通道**（`nChannels = 1`，即所有数据由一个通道处理）。
- **每个 chunk 大小：8 个元素**（`chunkCount = 8` 个元素，每个 chunk 为 32 字节）。

### **起始数据分布**

- 每个 GPU 初始有一份独立的数据：
  - GPU 0 的数据：`[A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]`
  - GPU 1 的数据：`[B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15]`

目标：完成 **AllReduce（sum）**，让每个 GPU 最终都得到：
```plaintext
[ A0+B0, A1+B1, ..., A15+B15 ]
```

---

### **数据划分**

1. **划分到通道：**
   - 只有一个通道（`nChannels = 1`），所有数据量（16 个元素）分配给这个通道。
   - **`channelCount = 16` 个元素**。

2. **划分到 chunk：**
   - 每个 chunk 处理 `chunkCount = 8` 个元素。
   - `channelCount = 16` 被划分为 2 个 chunk：
     - **Chunk 0：`[A0, A1, ..., A7]`**
     - **Chunk 1：`[A8, A9, ..., A15]`**

---

### **运行代码的细节**

#### 第一轮：`elemOffset = 0`，处理第 1 个 chunk

- **数据：Chunk 0（前 8 个元素）**
  - GPU 0：`[A0, A1, ..., A7]`
  - GPU 1：`[B0, B1, ..., B7]`

##### **(1) GPU 0 发送 chunk 0**
```cpp
prims.directSend(offset, offset, nelem);
```
- GPU 0 把 `[A0, A1, ..., A7]` 发送到 GPU 1。

##### **(2) GPU 1 接收并归约**
```cpp
prims.directRecvReduceDirectSend(offset, offset, nelem);
```
- GPU 1 接收 GPU 0 的数据 `[A0, A1, ..., A7]`，并和自己的数据 `[B0, B1, ..., B7]` 做归约（`sum`），结果为：
  ```plaintext
  [A0+B0, A1+B1, ..., A7+B7]
  ```
- GPU 1 将归约后的结果 `[A0+B0, A1+B1, ..., A7+B7]` 发送回 GPU 0。

##### **(3) GPU 0 接收归约结果**
```cpp
prims.directRecv(offset, offset, nelem);
```
- GPU 0 接收归约后的数据 `[A0+B0, A1+B1, ..., A7+B7]`，存入自己的接收缓冲区。

#### **第一轮完成，GPU 数据状态：**
- GPU 0：`[A0+B0, A1+B1, ..., A7+B7, A8, A9, ..., A15]`
- GPU 1：`[A0+B0, A1+B1, ..., A7+B7, B8, B9, ..., B15]`

---

#### 第二轮：`elemOffset = 8`，处理第 2 个 chunk

- **数据：Chunk 1（后 8 个元素）**
  - GPU 0：`[A8, A9, ..., A15]`
  - GPU 1：`[B8, B9, ..., B15]`

##### **(1) GPU 0 发送 chunk 1**
```cpp
prims.directSend(offset, offset, nelem);
```
- GPU 0 把 `[A8, A9, ..., A15]` 发送到 GPU 1。

##### **(2) GPU 1 接收并归约**
```cpp
prims.directRecvReduceDirectSend(offset, offset, nelem);
```
- GPU 1 接收 GPU 0 的数据 `[A8, A9, ..., A15]`，并和自己的数据 `[B8, B9, ..., B15]` 做归约（`sum`），结果为：
  ```plaintext
  [A8+B8, A9+B9, ..., A15+B15]
  ```
- GPU 1 将归约后的结果 `[A8+B8, A9+B9, ..., A15+B15]` 发送回 GPU 0。

##### **(3) GPU 0 接收归约结果**
```cpp
prims.directRecv(offset, offset, nelem);
```
- GPU 0 接收归约后的数据 `[A8+B8, A9+B9, ..., A15+B15]`，存入自己的接收缓冲区。

#### **第二轮完成，GPU 数据状态：**
- GPU 0：`[A0+B0, A1+B1, ..., A15+B15]`
- GPU 1：`[A0+B0, A1+B1, ..., A15+B15]`

---

### **结果验证**

经过两轮通信，每个 GPU 的数据已经完成了 AllReduce 操作，最终数据是：
```plaintext
[A0+B0, A1+B1, ..., A15+B15]
```

---

### **总结**

#### **`channelCount` 是什么？**
- `channelCount` 表示当前通道需要处理的数据总量（以数据元素为单位）。
- 在本例中：
  - 数据总量是 `16` 个 `float` 元素。
  - 因为只有 1 个通道，所以 `channelCount = 16`。

#### **`chunk` 是什么？**
- `chunk` 是通道内划分的最小数据单位。
- 在本例中：
  - 每个 `chunk` 包含 `8` 个 `float` 元素。
  - `channelCount = 16` 被划分为 2 个 `chunk`。

#### **通信的过程：**
1. 每个通道负责一部分数据（`channelCount`）。
2. 数据进一步被划分为多个 `chunk`，每次传输一个或多个 `chunk`。
3. 每个 `chunk` 在环形拓扑中依次传递，完成发送、接收、归约和最终存储。

这样划分的好处是：
- **并行化：** 通道级和 chunk 级的并行分工可以高效利用带宽。
- **负载均衡：** 每个 GPU 按 `chunk` 处理一部分数据，避免了单点瓶颈。

希望这个具体的例子可以帮助你完全理解 `channelCount` 和 `chunk` 的含义以及它们在 AllReduce 中的具体作用！


* directRecvReduceDirectSend
* recvReduceSend
* GenericOp
* directSend有两种：
    * prims_simple走的是:
    ```c
    __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  ```
    * ll和ll128走的是:
    ```c
    __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }

    __device__ void send(intptr_t inpIx, int eltN) {
    return GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  ```

* 调用reduceCopy的：
    * sendrecv.h
    * prims_simple.h
    * reduce_scatter.h
    * all_gather.h

* runRing里面调用了
    * prims.directSend(offset, offset, nelem);
    * prims.directRecvReduceDirectSend(offset, offset, nelem);
    * prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);
    * prims.directRecvCopyDirectSend(offset, nelem);
    * prims.directRecv(offset, offset, nelem);

* 情况一，采用prims_simple
    * directSend:
    ```c
    __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
    ```
    然后调用genericOp
    ```c
    template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem && ((flags & NetRegMode) == 0)) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      #if __CUDA_ARCH__ < 700
        // Above doesn't matter on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
        subBarrier();
        /* if user abort the kernel, we don't need to actually perform copy/reduce; just set size
         * to 0 to avoid unnecessary workload. */
        int workSize = ncclShmem.aborted ? 0 : sliceSize;
        if (flags & AnyNetDeviceUnpack) {
          ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
          // Sync here to make sure all workers are reading from the updated srcs)
          subBarrier();
        }

        if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
            /* NVLS can have srcs[0] == dsts[0], but we cannot enter this "if branch",
             * so we need to check whether MultimemSrcs and MultimemDsts are 0. */
            && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/0>
              (tid, nworkers, /*redArg*/0, /*preOpArgs*/nullptr, /*postOp*/false,
               1, ncclShmem.groups[group].srcs,
               fan.nsend(), ncclShmem.groups[group].dsts+1,
               workSize);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // For broadcast in CollNet to do empty send
          reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/0>
            (tid, nworkers, ncclShmem.redOpArgs[0],  nullptr, postOp,
             Recv, ncclShmem.groups[group].srcs,
             Dst, ncclShmem.groups[group].dsts,
             workSize);
        } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                    DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          reduceCopy<Unroll, RedOp, T,
            MultimemSrcs, Recv+Src, Recv*MaxRecv+Src,
            MultimemDsts, Send+Dst, Send*MaxSend+Dst, PreOpSrcs>
            (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
             Recv*fan.nrecv()+Src, ncclShmem.groups[group].srcs,
             Send*fan.nsend()+Dst, ncclShmem.groups[group].dsts,
             workSize);
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < sliceSize);
        offset += sliceSize;
        slice += 1;
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, 0);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      postPeer<Recv, Send>(0 < sliceSize);
      offset += sliceSize;
      slice += 1;
    }
  }
    ```
    然后里面directSend的条件下，调用了reduceCopy，进而调用reduceCopyPacks
