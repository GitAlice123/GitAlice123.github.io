---
title: NCCL代码阅读-06
date: 2025-01-03 11:23:23
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---
## RunWorkColl
* 书接上文。RunWorkBatch走到了这里：
```c
RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work)
```
* 其中work是一个ncclDevWorkColl
* 然后按照模版实例化的参数（这边以LL128为例），进入了这个函数：
```c
template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};
```

## runRing
```c
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int ringIx = ring->index;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t gridOffset;
    ssize_t channelCount;
    ssize_t chunkCount;
    // gridOffset:
    // channelCount:当前通道要负责的总的元素数量
    // chunkCount: 当前通道中一个chunk容纳的元素数量
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nranks * chunkCount;
    ssize_t offset;
    int nelem;
    int chunk;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);


    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      ssize_t chunkOffset;

      if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nranks), 16/sizeof(T));

      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      // step 0: push data to next GPU
      chunk = modRanks(ringIx + nranks - 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directSend(offset, offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j = 2; j < nranks; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvReduceDirectSend(offset, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ringIx + 0;
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);
      prims.directRecvReduceCopyDirectSend(offset, offset, nelem, /*postOp=*/true);

      // k-2 steps: copy to next GPU
      for (int j = 1; j < nranks - 1; ++j) {
        chunk = modRanks(ringIx + nranks - j);
        chunkOffset = chunk * chunkCount;
        offset = gridOffset + elemOffset + chunkOffset;
        nelem = (int)min(chunkCount, remCount - chunkOffset);
        prims.directRecvCopyDirectSend(offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = modRanks(ringIx + 1);
      chunkOffset = chunk * chunkCount;
      offset = gridOffset + elemOffset + chunkOffset;
      nelem = (int)min(chunkCount, remCount - chunkOffset);

      prims.directRecv(offset, offset, nelem);
    }
  }
```
* 这个函数里面东西比较多，我们一点点拿出来看
* 下面解释ncclCollCdbPart，prims的初始化，directSend
### ncclCollCbdPart
```c
template<typename Int>
__host__ __device__ inline void ncclCollCbdPart(
    struct ncclDevWorkColl* work, uint32_t channelId, int proto, int eltSize,
    Int* count, Int* partOffset, Int* partCount, Int* chunkCount
  ) {
    /////////////////////////////////////////////
    // proto: LL128
    // eltSize: sizeof(uint64_t)
    // count: 我们的例子是nullptr
    /////////////////////////////////////////////

    // 一个grain有几个数据元素
  int eltPerGrain = ncclProtoGrainSize(proto)/eltSize;
  int nMidChannels = work->channelHi - work->channelLo - 1;
  // We can assum that nMidChannels<0 implies countMid==0, which let's us assume
  // that countMid*nMidChannels == 0.
  if (count != nullptr) {
    *count = work->cbd.countLo + work->cbd.countMid*nMidChannels + work->cbd.countHi;
  }
  if (channelId == work->channelLo) {
    *partOffset = 0;
    // countLo是当前通道要负责的总的元素数量
    *partCount = work->cbd.countLo;
    // chunkGrainsLo: 低通道上的一个chunk可以容纳几个grain
    // chunkCount: 当前通道上一个chunk里面的元素数量
    *chunkCount = work->cbd.chunkGrainsLo*eltPerGrain;
  } else if (channelId == work->channelHi) {
    *partOffset = work->cbd.countLo + nMidChannels*work->cbd.countMid;
    *partCount = work->cbd.countHi;
    *chunkCount = work->cbd.chunkGrainsHi*eltPerGrain;
  } else {
    int mid = channelId - work->channelLo - 1;
    *partOffset = work->cbd.countLo + mid*work->cbd.countMid;
    *partCount = work->cbd.countMid;
    *chunkCount = work->cbd.chunkGrainsMid*eltPerGrain;
  }
}
```
### primitives里面有哪些东西
#### 类内变量
```c
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;// 由传递过来的Fan决定
  static constexpr int Input=0, Output=1;
  RedOp redOp;
  const int tid; // thread index in primitives group
  const int nthreads; // 调用prims的函数里面可以看到，这是执行这个work需要的总线程数（work里面确定了）
  const int wid; // 在当前warp中线程的编号
  const int stepSize;
  const int warp; // 当前是第几个warp
  const int warpInBlock; // 在这个线程块里面，这是第几个warp
  const bool flagThread;
  const int group;
  Fan fan;
  T *userBufs[2];// 两个指针，指向的类型是T
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile struct ncclConnFifo* sendConnFifo = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[MaxRecv];
  uint64_t sendStep[MaxSend];
  uint64_t* recvBuff[MaxRecv];// 存接收缓冲区地址的数组
  uint64_t* sendBuff[MaxSend];// 存发送缓冲区地址的数组
```
* 这里大概是一个怎样的结构呢？我梳理了一下：
    * 我们假设有A,B,C三个节点
    * A和B，A和C通信使用的是同一个channel，我这边觉得翻译成频段是更有助于理解的
    * channel（实际类里面应该是ncclDevChannel）里面有一个ncclDevChannelPeer的指针数组peers，也就是说，channel.peers[0]是指向一个ncclDevChannelPeer（比如B节点）的指针，channel.peers[1]是指向另一个ncclDevChannelPeer（比如C节点）的指针
    * 
### prims的初始化
* 在runRing里面，对prims进行了初始化：
```c
Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
    (tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
```
* 寻找到对应的构造函数，在prims_ll128里：
```c
  __device__ Primitives(
      const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv=0, uint8_t connIndexSend=0, struct ncclDevWorkColl* e = nullptr,
      bool ipcReg = false, bool netReg = false, int stepSize_ = 0
    ):
    redOp(redOpArg),
    // wid是在当前warp中线程的编号
    // warp是当前是第几个warp
    tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE),
    // warpInBlock是当前线程在当前线程块内属于第几个Warp
    warpInBlock(threadIdx.x/WARP_SIZE),
    flagThread((tid%8)==7), group(group),
    // 步长是多少个元素
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS/sizeof(uint64_t)) {
    auto *channel = &ncclShmem.channel;
    // 最初初始化时使用的是对称的通信模式(FanSymmetric)，即接收和发送的对端节点数量是相等的
    // nrecv和nsend就是实际接收和发送的对端节点数量，MaxRecv是在初始化的时候FanSymmetric<1>指定的
    int nrecv=0, nsend=0;
    while (nrecv < MaxRecv && recvPeers[nrecv] >= 0) {
      loadRecvConn(&channel->peers[recvPeers[nrecv]]->recv[connIndexRecv], nrecv);
      nrecv++;
    }
    while (nsend < MaxSend && sendPeers[nsend] >= 0) {
      loadSendConn(&channel->peers[sendPeers[nsend]]->send[connIndexSend], nsend);
      nsend++;
    }
    this->fan = Fan(nrecv, nsend);
    // Coverity reports recvConn and sendConn being possibly NULL at this point but that won't actually
    // happen given the two "while" loops just above.
    // coverity[var_deref_model:FALSE]
    loadRecvSync();
    // coverity[var_deref_model:FALSE]
    loadSendSync();
    setDataPtrs(inputBuf, outputBuf);
  }

```
#### loadRecvConn
```c
__device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    // 这里conn->buffs的值，即接收缓冲区（以及发送缓冲区）的地址，是在p2p.cc里面，初始化的时候就填写好的，接收缓冲区的地址是
    recvBuff[i] = (union ncclLLFifoLine*)conn->buffs[NCCL_PROTO_LL];
    recvStep[i] = conn->step;
    if (wid == i) recvConn = conn;
}
```
### prims.directSend
* 在中间的几个步骤里，执行了
```c
prims.directSend(offset, offset, nelem);
```
* 一路往下找，在primitives里面：
```c
template<typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  //...
}
```
* 在prims_ll128里面：
```c
  __device__ void send(intptr_t inpIx, int eltN) {
    return GenericOp<0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
```
### GenericOp
```c
  template <int RECV, int SEND, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void GenericOp(intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp) {
    //////////////////////////////////////////
    // RECV=0, SEND=1, SrcBuf=Input=0, DstBuf=0
    // srcIx=inpIx(offset), dstIx=-1, nelem=eltN, postOp=false
    //////////////////////////////////////////
    constexpr int SRC = SrcBuf != -1 ? 1 : 0;//1
    constexpr int DST = DstBuf != -1 ? 1 : 0;//0
    T const *srcPtr = SrcBuf == -1 ? nullptr : userBufs[SrcBuf] + srcIx;
    T       *dstPtr = DstBuf == -1 ? nullptr : userBufs[DstBuf] + dstIx;
    int wireOffset = WireWordPerSlice*warp + 2*wid;
    const int nwarps = nthreads/WARP_SIZE;
    nelem = nelem < 0 ? 0 : nelem;

    // 控制路径，是否时间很长?
    if (SEND) waitSend(divUp(nelem, DataEltPerSlice)*WireWordPerSlice*sizeof(uint64_t));
    barrier();
    nelem -= DataEltPerSlice*warp;
    srcPtr += DataEltPerSlice*warp;
    dstPtr += DataEltPerSlice*warp;
    while (nelem > 0) {
      const int eltInSlice = min(nelem, DataEltPerSlice);
      uint64_t regs[NCCL_LL128_SHMEM_ELEMS_PER_THREAD];
      if (SRC) loadRegsBegin(regs, srcPtr, eltInSlice);
      recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SrcBuf, DstBuf>(regs, wireOffset, postOp);
      if (DST) storeRegs(dstPtr, regs, eltInSlice);

      wireOffset += WireWordPerSlice*nwarps;
      srcPtr += DataEltPerSlice*nwarps;
      dstPtr += DataEltPerSlice*nwarps;
      nelem -= DataEltPerSlice*nwarps;
    }

    barrier();
    if (SEND) for (int i=0; i < MaxSend; i++) sendStep[i] += 1;
    if (SEND) postSend();
    if (RECV) for (int i=0; i < MaxRecv; i++) recvStep[i] += 1;
    if (RECV) postRecv();
  }

```