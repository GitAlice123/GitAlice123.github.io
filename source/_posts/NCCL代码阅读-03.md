---
title: NCCL代码阅读-03
date: 2024-12-9 17:57:36
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---

## 通信组创建和销毁(官网给的例子，解释看注释)
### 一个进程，一个线程，多个设备
* 在这种单进程的场景下，可以使用ncclCommInitAll()
```c
int main(int argc, char *argv[])
{
    ncclComm_t comms[4];

    // managing 4 devices
    int nDev = 4;
    int size = 32 * 1024 * 1024;
    int devs[4] = {0, 1, 2, 3};

    // 这部分的解释见《NCCL代码阅读-02》的cudaAlloc部分
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


## 一个具体的调用路径：sendrecv操作

* 不得不说nccl的调用真是够复杂的......
* 我们就从nccl-test（官方给的测试代码）入手，看看sendrecv这个最简单的操作是怎么做的
### SendRecvRunColl
* 这个函数是nccl-test里面的一个测试函数，用于测试sendrecv操作
```c
testResult_t SendRecvRunColl(void *sendbuff, void *recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
{
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    int rank;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    int recvPeer = (rank - 1 + nRanks) % nRanks;
    int sendPeer = (rank + 1) % nRanks;

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sendbuff, count, type, sendPeer, comm, stream));
    NCCLCHECK(ncclRecv(recvbuff, count, type, recvPeer, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
}
```
* 这里可以很明显的看到，接收方和发送方，都要显式调用一次操作，ncclSend和ncclRecv
* ncclSend和ncclRecv被包裹在了ncclGroupStart和ncclGroupEnd里面

### ncclSend和ncclRecv（以ncclSend为例）
```c
ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream)
{
    NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
    NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

    struct ncclInfo info = {ncclFuncSend, "Send",
                            NULL, (void *)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
                            1, 1};
    ncclResult_t ret;
    NCCLCHECK(ncclGroupStart());
    NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
    NCCLCHECK(ncclGroupEnd());
    return ret;
}
```
* 这边可以看到，在ncclSend里面有一个很重要的数据结构**ncclInfo**，这个结构体里面包含了这次通信的所有信息，其中**操作种类**是被第一个参数传进去的，ncclInfo这个数据结构的介绍见《NCCL代码阅读-02》
### ncclEnqueueCheck
```c
ncclResult_t ncclEnqueueCheck(struct ncclInfo *info)
{
    NCCLCHECK(ncclGroupStartInternal());
    ncclResult_t ret = ncclSuccess;
    int devOld = -1;

    NCCLCHECKGOTO(CommCheck(info->comm, info->opName, "comm"), ret, fail);
    // Check whether communicator is ready to communicate
    NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

    if (info->comm->checkPointers)
    {
        CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
        CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
    }
    NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

    INFO(NCCL_COLL, "%s: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p",
         info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
         info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);
    TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zu,%d,%d,%d,%p,%p)", info->opName, reinterpret_cast<int64_t>(info->sendbuff), reinterpret_cast<int64_t>(info->recvbuff), info->count, info->datatype, info->op, info->root, info->comm, info->stream);

    NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

exit:
    if (devOld != -1)
        CUDACHECK(cudaSetDevice(devOld));
    ncclGroupErrCheck(ret);
    NCCLCHECK(ncclGroupEndInternal());
    /* if depth is 1, ncclGroupEndInternal() will trigger group ops. The state can change
     * so we have to check state here. */
    if (info->comm && !info->comm->config.blocking)
    {
        NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret));
    }
    return ret;
fail:
    if (info->comm && !info->comm->config.blocking)
        (void)ncclCommSetAsyncError(info->comm, ret);
    goto exit;
}
```
* 前面就是做了一些入队的检查，真正进行入队操作的是**taskAppend**函数
* taskAppend函数将info转换成了一个task，并且将这个task放入对应comm的comm->planner中，这个planner，即ncclKernelPlanner，是一个比较复杂的数据结构，简要来说就是一个comm上任务的调度器，这个数据结构的介绍后续会放入《NCCL代码阅读-02》
* 在**ncclGroupEndInternal**里面，调用了**groupLaunch**，**groupLaunch**中的**doLaunches**调用了**ncclLaunchKernel(comm, plan)**，这个函数就是真正的调用CUDA kernel的地方

### ncclLaunchKernel
```c
ncclResult_t ncclLaunchKernel(struct ncclComm *comm, struct ncclKernelPlan *plan)
{
    struct ncclKernelPlanner *planner = &comm->planner;
    int nChannels = countOneBits(plan->channelMask);
    void *sym = plan->kernelFn;
    dim3 grid = {(unsigned)nChannels, 1, 1};
    dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
    int smem = ncclShmemDynamicSize(comm->cudaArch);
    cudaStream_t launchStream = planner->streams->stream;
    void *extra[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
        CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
        CU_LAUNCH_PARAM_END};

    CUfunction fn;
    CUDACHECK(cudaGetFuncBySymbol(&fn, sym));

#if CUDART_VERSION >= 11080
    int driverVersion;
    NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
    if (driverVersion >= 11080)
    {
        int compCap = comm->compCap;
        unsigned int clusterSize = (compCap == 90) ? comm->config.cgaClusterSize : 0;

        CUlaunchConfig launchConfig = {0};
        CUlaunchAttribute launchAttrs[3];
        int attrs = 0;
        /* Cooperative Group Array (CGA)
         * On sm90 and later we have an extra level of hierarchy where we
         * can group together several blocks within the Grid, called
         * Thread Block Clusters.
         * Clusters enable multiple thread blocks running concurrently
         * across multiple SMs to synchronize and collaboratively fetch
         * and exchange data. A cluster of blocks are guaranteed to be
         * concurrently scheduled onto a group of SMs.
         * The maximum value is 8 and it must be divisible into the grid dimensions
         */
        if (clusterSize)
        {
            // Grid dimension must be divisible by clusterSize
            if (grid.x % clusterSize)
                clusterSize = 1;
            launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
            launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
            launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
            launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
        }
#if CUDART_VERSION >= 12000
        if (compCap >= 90 && driverVersion >= 12000)
        {
            // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
            launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
            launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain)ncclParamMemSyncDomain();
        }
#endif
        launchConfig.gridDimX = grid.x;
        launchConfig.gridDimY = grid.y;
        launchConfig.gridDimZ = grid.z;
        launchConfig.blockDimX = block.x;
        launchConfig.blockDimY = block.y;
        launchConfig.blockDimZ = block.z;
        launchConfig.sharedMemBytes = smem;
        launchConfig.attrs = launchAttrs;
        launchConfig.numAttrs = attrs;
        launchConfig.hStream = launchStream;

        // CUDACHECK(cudaLaunchKernelExC(&launchConfig, fnAddr, args));
        CUCHECK(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra));
        return ncclSuccess;
    }
#endif
    // Standard kernel launch
    CUCHECK(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra));
    // CUDACHECK(cudaLaunchKernel(fnAddr, grid, block, args, smem, launchStream));
    return ncclSuccess;
}
```
* 这里面最后调用的是cuLaunchKernel，其中fn是通过cudaGetFuncBySymbol获取的，这个symbol就是nccl的kernel函数
* 这个sym来自plan->kernelFn，它在**scheduleP2pTasksToPlan**中被赋值:
```c
    if (!plan->kernelSpecialized)
    {
        plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
        plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
    }
```
* 其中ncclDevFunId_P2p()是这样的：
```c
inline int ncclDevFuncId_P2p() { return ncclDevFuncRowToId[0]; }
```
* 这个ncclDevFuncRowToId是一个映射表，填写这个映射表的位置还挺难找，在nccl/src/device/下面，有一个**generate.py**，会在build里面生成一个nccl/build/obj/device/gensrc/host_table.cc
* 那CUDA是怎么通过fn去找到对应的kernel函数的呢？我们仍然要看generate.py，这个脚本还会生成一组文件，其中一个是nccl/build/obj/device/gensrc/sendrecv.cu，这里面的内容是这样的：
```c
#include "common.h"
#include "sendrecv.h"
DEFINE_ncclDevKernel(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 589)
DEFINE_ncclDevFunc(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)
```
* 这里面的两个宏定义在nccl/src/device/common.h里面：
```c
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId)                    \
    __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K)        \
    {                                                                                                  \
        ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
    }

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
    __device__ void ncclDevFunc_##suffix()                       \
    {                                                            \
        RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run();  \
    }
```
* CUDA会通过查找刚刚的**host_table.cc**找到这个**ncclDevKernel_SendRecv**，然后通过这个函数去调用真正的kernel函数（去文件里面搜一下"ncclDevKernel_SendRecv"看一下就大概知道了）
* 下面我们看看RunWorkBatch是什么东西

### RunWorkBatch
```c
template <ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch;
```
* 这是他的最初原型，在nccl/src/device/sendrecv.h里面，我们可以看到他的一个针对sendrecv的特化：
```c
template <typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>
{
    static_assert(sizeof(T) == 1, "SendRecv only works on single byte types T.");

    template <typename Proto>
    __device__ void runSend(int tid, int tn, int group, struct ncclDevWorkP2p *work)
    {
        size_t bytes = work->sendBytes;
        int chunkSize = work->sendIpcReg && ncclShmem.comm.isNvlink ? (1 << 30) : u32fp8Decode(work->sendChunkSize_u32fp8);
        Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1>
            prims(tid, tn, nullptr, &work->sendRank, work->sendAddr, nullptr,
                  /*redOpArg(ignored)=*/0, group, 1, 1, nullptr,
                  /*ipcReg=*/work->sendIpcReg, /*netReg=*/work->sendRegistered, ncclShmem.comm.p2pChunkSize);
        size_t cursor = 0;
        do
        {
            int n = min(size_t(chunkSize), bytes - cursor);
            prims.directSend(cursor, cursor, n);
            cursor += n;
        } while (cursor < bytes && work->sendRegistered == 0);
    }

    template <typename Proto>
    __device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p *work)
    {
        size_t bytes = work->recvBytes;
        int chunkSize = work->recvIpcReg && ncclShmem.comm.isNvlink ? (1 << 30) : u32fp8Decode(work->recvChunkSize_u32fp8);
        Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
            prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr,
                  /*redOpArg(ignored)=*/0, group, 1, 1, nullptr,
                  /*ipcReg=*/work->recvIpcReg, /*netReg=*/work->recvRegistered, ncclShmem.comm.p2pChunkSize);
        size_t cursor = 0;
        do
        {
            int n = min(size_t(chunkSize), bytes - cursor);
            prims.directRecv(cursor, cursor, n);
            cursor += n;
        } while (cursor < bytes && work->recvRegistered == 0);
    }

    __device__ __forceinline__ void run()
    {
        const int tid = threadIdx.x;
        const int tn = blockDim.x;
        const int wid = tid / WARP_SIZE;
        const int nWarps = tn / WARP_SIZE;
        const int lane = tid % WARP_SIZE;

        struct Shared
        {
            uint32_t workSendMask; // bitmasks of which work indices have send/recv
            uint32_t workRecvMask;
        };
        Shared *shared = (Shared *)ncclScratchForWarp(0);

        struct ncclDevWorkP2p *works = (ncclDevWorkP2p *)ncclShmem.workStorage;
        int nWorks = ncclShmem.nWorks;

        if (wid == 0)
        {
            // Modify the memory range of each work[] to reflect this channel's
            // partition of the work. Since integer divides are very heavy it's
            // best to do them all in one warp.
            int workIx = lane % 16;
            int isSend = lane < 16 ? 0 : 1;
            bool hasWork = false;
            if (workIx < nWorks)
            {
                struct ncclDevWorkP2p *work = &works[workIx];
                size_t bytes = isSend ? work->sendBytes : work->recvBytes;
                int nParts = isSend ? work->nSendChannels : work->nRecvChannels;
                int part = ncclP2pChannelToPart(work->nP2pChannels, work->channelBase, ncclShmem.channelId);
                hasWork = (part < nParts);
                if (nParts != 0)
                {
                    size_t partBeg, partEnd;
                    ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
                    (isSend ? work->sendAddr : work->recvAddr) = (char *)(isSend ? work->sendAddr : work->recvAddr) + partBeg;
                    (isSend ? work->sendBytes : work->recvBytes) = partEnd - partBeg;
                }
            }
            // Coverity reports a possible thread divergence due to not all threads participating in the collective.
            // However, the code ensures that the participation is on a per-warp basis.
            // coverity[device_thread_diverged:FALSE]
            uint32_t mask = __ballot_sync(~0u, hasWork);
            if (lane == 0)
            {
                shared->workSendMask = mask >> 16;
                shared->workRecvMask = mask & 0xffff;
            }
        }

        // The fastest way to compute a warp uniform division x/y in [0,32) is to
        // use each lane to guess a solution and count the ones that don't exceed
        // the numerator:
        //   __popc(__ballot_sync(~0u, y*(lane+1) <= x))
        // That takes 1/3 the time of standard division and about 3/4 the time of
        // approximate floating point division:
        //   __float2int_rd(__fdividef(float(x),float(y))).

        // nWarpPerWork = nWarps/nWorks
        int nWarpPerWork = __popc(__ballot_sync(~0u, nWorks * (lane + 1) <= nWarps));
        int nRecvWarpPerWork = nWarpPerWork <= 4 ? nWarpPerWork / 2 : (nWarpPerWork - 1) / 2;
        int nSendWarpPerWork = nWarpPerWork <= 4 ? nRecvWarpPerWork : nRecvWarpPerWork + 1;
        // This might reduce nWarpPerWork which is probably desirable. It is better
        // to have a balanced number of reading and writing threads even if that
        // leaves warps unused.
        nWarpPerWork = nSendWarpPerWork + nRecvWarpPerWork;
        // The work index this warp belongs to: workIx = wid/nWarpPerWork
        int workIx = __popc(__ballot_sync(~0u, (lane + 1) * nWarpPerWork <= wid));

        __syncthreads(); // Wait for works[] and shared->* to be updated by warp=0

        uint32_t workSendMask = shared->workSendMask;
        uint32_t workRecvMask = shared->workRecvMask;

        __syncthreads(); // release scratch space used by shared->*
        if (nWorks <= workIx)
            return;

        // Thread range for whole work (send & recv combined)
        int subtid = tid - workIx * nWarpPerWork * WARP_SIZE;
        int subtn = nWarpPerWork * WARP_SIZE;

        // A send primtive of sufficient size requires 2 cuda barrier ids.
        constexpr int nSendWarpsForExtraGroup = NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE / WARP_SIZE;
        // Count up all group ids used below this workIx:
        int group, extra;
        // Each recv gets one group id:
        group = __popc(workRecvMask & ((1 << workIx) - 1));
        // Sends accompanying recvs get one and maybe an extra:
        extra = (nSendWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
        group += __popc((workSendMask & workRecvMask) & ((1 << workIx) - 1)) * (1 + extra);
        // Sends without recvs use more warps so compute extra accordingly:
        extra = (nWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
        group += __popc((workSendMask & ~workRecvMask) & ((1 << workIx) - 1)) * (1 + extra);

        struct ncclDevWorkP2p *work = &works[workIx];
        bool hasSend = 1 & (workSendMask >> workIx);
        bool hasRecv = 1 & (workRecvMask >> workIx);
        bool isCopy = work->sendRank == ncclShmem.comm.rank;
        bool isSend = !hasRecv || (hasSend && subtid < nSendWarpPerWork * WARP_SIZE);

        if (!isCopy && hasSend && hasRecv)
        {
            // Translate thread ids to reflect just this send or recv as opposed to whole work.
            if (isSend)
            {
                subtn = nSendWarpPerWork * WARP_SIZE;
            }
            else
            {
                subtid -= nSendWarpPerWork * WARP_SIZE;
                subtn = nRecvWarpPerWork * WARP_SIZE;
                group += 1 + (nSendWarpPerWork >= nSendWarpsForExtraGroup ? 1 : 0);
            }
        }

        if (isCopy)
        {
            reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>(subtid, subtn, 0, nullptr, false, 1, &work->sendAddr, 1, &work->recvAddr, (ssize_t)work->sendBytes);
        }
        else if (isSend)
        {
            if (work->sendProtoLL)
            {
                runSend<ProtoLL>(subtid, subtn, group, work);
            }
            else
            {
                runSend<ProtoSimple<1, 1>>(subtid, subtn, group, work);
            }
        }
        else
        {
            if (work->recvProtoLL)
            {
                runRecv<ProtoLL>(subtid, subtn, group, work);
            }
            else
            {
                runRecv<ProtoSimple<1, 1>>(subtid, subtn, group, work);
            }
        }
    }
};
```
* run里面调用了runSend,runRecv，里面调用了primitives原语，接下来可以前往nccl/src/device/prims_ll128.h等文件查看相关内容