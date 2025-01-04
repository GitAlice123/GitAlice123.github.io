---
title: NCCL代码阅读-01
date: 2024-11-28 15:45:03
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---

## nccl源码结构

### 整体结构概览
```shell
.
├── bootstrap.cc
├── channel.cc
├── collectives.cc
├── debug.cc
├── device
│   ├── all_gather.h
│   ├── all_reduce.h
│   ├── broadcast.h
│   ├── common.cu
│   ├── common.h
│   ├── common_kernel.h
│   ├── generate.py
│   ├── Makefile
│   ├── network
│   │   └── unpack
│   │       ├── unpack_defs.h
│   │       └── unpack.h
│   ├── onerank.cu
│   ├── op128.h
│   ├── primitives.h
│   ├── prims_ll128.h
│   ├── prims_ll.h
│   ├── prims_simple.h
│   ├── reduce.h
│   ├── reduce_kernel.h
│   ├── reduce_scatter.h
│   └── sendrecv.h
├── enhcompat.cc
├── enqueue.cc
├── graph
│   ├── connect.cc
│   ├── paths.cc
│   ├── rings.cc
│   ├── rings.h
│   ├── search.cc
│   ├── topo.cc
│   ├── topo.h
│   ├── trees.cc
│   ├── tuning.cc
│   ├── xml.cc
│   └── xml.h
├── group.cc
├── include
│   ├── 略
├── init.cc
├── init_nvtx.cc
├── Makefile
├── misc
│   ├── argcheck.cc
│   ├── cudawrap.cc
│   ├── gdrwrap.cc
│   ├── ibvsymbols.cc
│   ├── ibvwrap.cc
│   ├── ipcsocket.cc
│   ├── nvmlwrap.cc
│   ├── param.cc
│   ├── profiler.cc
│   ├── shmutils.cc
│   ├── socket.cc
│   ├── strongstream.cc
│   ├── tuner.cc
│   └── utils.cc
├── nccl.h.in
├── nccl.pc.in
├── net.cc
├── proxy.cc
├── register.cc
├── transport
│   ├── coll_net.cc
│   ├── generic.cc
│   ├── net.cc
│   ├── net_ib.cc
│   ├── net_socket.cc
│   ├── nvls.cc
│   ├── p2p.cc
│   └── shm.cc
└── transport.cc
```
### device
* device目录里面就是给GPU执行的代码了，可以看到里面函数前面的__device__等标志
#### common.h
* 定义了ncclShmemGroup和ncclShmemData结构体
* 定义了**__shared__ ncclShmemData ncclShmem**，shared修饰表示是GPU一个block中所有thread的共享内存
* 一些用于同步的汇编代码
* 定义了copyToShmem16函数，用于将一些数据（其实不是主要数据搬运，代码里可以看到，主要是用于搬运comm和channel这些控制信息的）从GPU的全局内存搬运到共享内存里
* 定义了loadWorkBatchToShmem函数，用于从全局内存加载工作批次到共享内存中。这些批次由 ncclShmem 中的 workStorage 存储，每个线程负责加载不同的部分。通过 tid 和 tn，每个线程计算自己需要处理的工作项，并将其加载到共享内存中
* 声明了RunWorkColl结构体，实际执行一个具体的集体操作计算
* 声明了专门用于P2P操作的RunWorkBatch结构体，在sendrecv.h里面实现
* 定义了通用的RunWorkBatch结构体，划分一下本线程要负责的work，然后调用RunWorkColl实际处理、
* 定义了ncclKernelMain函数，主函数，初始化该进程块负责哪个通道，然后执行每个批次的集体操作
* 宏定义DEFINE_ncclDevKernel和DEFINE_ncclDevFunc，见NCCL代码阅读-03


## 创建一个通信组(communicator)                                                 
* 创建一个通信组之前，每个CUDA设备都要被分配一个唯一的rank id
* 有了这个rank id和CUDA设备的静态映射，ncclCommInitRank(), ncclCommInitRankConfig() and ncclCommInitAll() 三个函数会创建communicator objects，每个communicator object会和一个固定的rank（及一个CUDA设备）关联。
* 在调用ncclCommInitRank之前，需要调用ncclGetUniqueId()来获取一个unique id，这个ID必须广播到所有参与通信的进程，让他们知道自己在communicator中
* 比如有四个GPU互相通信，加入了一个通信组，那么这个通信组就需要一个通信上下文记录所有的信息
* 类比四个人开会，那么这个通信上下文就是会议室

### ncclCommInitRank
```c
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
```
* 创建一个communicator object
* 里面调用ncclCommInitRankDev()

### ncclCommInitAll
* 在**一个CPU进程**里面执行(**因此他后面所调用的所有函数都是在这一个进程，一个线程里面执行的**)，创建多个communicator object
* 但是只能是单进程版本，也因此不支持多node通信
* 首先检查了各种数据的有效性
* 然后调用ncclGetUniqueId()获取一个unique id
    * ncclGetUniqueId()首先调用ncclInit()初始化NCCL

### ncclInit()
* 这是一个在所有线程中只会执行一次的函数
* 在两个地方被调用：ncclGetUniqueId和ncclCommInitRankDev
* 如果是ncclGetUniqueId调用的，那么分两种情况：
    * 在ncclCommInitAll中调用，那其实就一个进程，一个线程，不用担心会被多次调用
    * 在ncclCommInitRank前面调用，那么就要限制只有第一个线程调用，后面的线程不会再调用

## SendRecv的调用流程
* nccl-test中，sendrecv.cu
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
    // 显式对sendPeer和recvPeer进行send和recv操作
    NCCLCHECK(ncclSend(sendbuff, count, type, sendPeer, comm, stream));
    NCCLCHECK(ncclRecv(recvbuff, count, type, recvPeer, comm, stream));
    NCCLCHECK(ncclGroupEnd());
    return testSuccess;
}
```
* nccl的collectives.cc中，注意在info里面传递的第一个参数是comm，也就是操作类型，后续在enqueucheck里面把操作类型用info传进去了
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
* nccl的enqueue.cc中
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

    // ！！！！！！！！！！！！！！！！！！！！！！！！！！
    // taskAppend把info转换成一个task，然后加入到comm->planner
    // 调用 ncclGroupCommJoin 将当前任务加入线程本地的通信组
    // 从内存池中分配一个 P2P 任务结构 (ncclTaskP2p) 并初始化任务
    // 将 P2P 任务添加到对应 peer 的发送队列或接收队列
    // 任务放入队列后，下面的groupEndInternal可见
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

* nccl的group.cc中
```c
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t *simInfo)
{
    ncclResult_t ret = ncclSuccess;
    ncclSimInfo_t internalSimInfo = NCCL_SIM_INFO_INITIALIZER;
    ncclSimInfo_t *internalSimInfoPtr = NULL;
    size_t realSize = 0;

    internalSimInfo.magic = 0;

    if (ncclGroupDepth == 0)
    {
        WARN("ncclGroupEnd: not in a group call.");
        ret = ncclInvalidUsage;
        goto exit;
    }

    if ((--ncclGroupDepth) > 0)
        goto exit;

    if ((ret = ncclGroupError) != ncclSuccess)
        goto fail;

    if (simInfo)
    {
        memcpy((void *)&realSize, (void *)&simInfo->size, sizeof(size_t));
        realSize = realSize > sizeof(ncclSimInfo_t) ? sizeof(ncclSimInfo_t) : realSize;
        memcpy((void *)&internalSimInfo, (void *)simInfo, realSize);
        if (internalSimInfo.magic != 0x74685283)
        {
            WARN("ncclSimInfo_t argument not initialized via NCCL_SIM_INFO_INITIALIZER");
            ret = ncclInvalidArgument;
            goto fail;
        }
        internalSimInfoPtr = &internalSimInfo;
    }

    if (ncclGroupCommHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs) || ncclGroupCommPreconnectHead != nullptr)
    {
        ncclGroupJobMain.groupCommHeadPtr = &ncclGroupCommHead;
        ncclGroupJobMain.groupCommPreconnectHeadPtr = &ncclGroupCommPreconnectHead;
        ncclGroupJobMain.groupErrorPtr = &ncclGroupError;
        ncclGroupJobMain.asyncJobsPtr = &ncclAsyncJobs;
        ncclGroupJobMain.abortFlagPtr = &ncclGroupJobAbortFlag;
        ncclGroupJobMain.groupBlockingPtr = &ncclGroupBlocking;
        ncclGroupJobMain.initialized = true;
        ncclGroupJobMainPtr = &ncclGroupJobMain;
        /* make sure ncclGroupBlocking has been set. */
        assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1);
        if (ncclGroupBlocking == 0)
        {
            /* nonblocking group */
            if (!ncclIntruQueueEmpty(&ncclAsyncJobs))
            {
                ncclAsyncJob *job = ncclIntruQueueHead(&ncclAsyncJobs);
                do
                {
                    NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
                    job->comm->groupJob = ncclGroupJobMainPtr;
                    job = job->next;
                } while (job);
            }

            if (ncclGroupCommHead)
            {
                ncclComm_t comm = ncclGroupCommHead;
                do
                {
                    NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
                    /* link group job to communicators. */
                    comm->groupJob = ncclGroupJobMainPtr;
                    comm = comm->groupNext;
                } while (comm);
            }

            ncclGroupJobMainPtr->base.func = groupLaunchNonBlocking;
            PTHREADCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void *)&ncclGroupJobMainPtr->base), "pthread_create", ret, fail);
            ret = ncclInProgress;
        }
        else
        {
            /* blocking group */
            NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base, internalSimInfoPtr), ret, fail);
            if (simInfo)
                memcpy((void *)simInfo, (void *)internalSimInfoPtr, realSize);
            groupResetJobState(ncclGroupJobMainPtr);
        }
    }

exit:
    return ret;
fail:
    groupCleanup(&ncclGroupCommHead, &ncclGroupCommPreconnectHead, &ncclAsyncJobs, &ncclGroupError, &ncclGroupBlocking, &ncclGroupJobAbortFlag, ret);
    goto exit;
}
```
* group.cc中
```c
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t *simInfo = NULL)
{
    int savedDev;
    ncclResult_t ret = ncclSuccess;
    struct ncclGroupJob *gjob = (struct ncclGroupJob *)job_;
    struct ncclComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
    struct ncclComm *groupCommPreconnectHeadMain = *gjob->groupCommPreconnectHeadPtr;
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = gjob->asyncJobsPtr;

    bool *groupAbortFlag = gjob->abortFlagPtr;

    CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);

    if (!simInfo && groupCommPreconnectHeadMain != nullptr)
    {
        struct ncclComm *comm = groupCommPreconnectHeadMain;
        do
        {
            struct ncclPreconnectJob *job;
            NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
            job->base.func = ncclP2PPreconnectFunc;
            job->base.undo = nullptr;
            job->base.destructor = free;
            job->base.state = ncclGroupJobRunning;
            job->base.abortFlag = comm->abortFlag;
            job->base.abortFlagDev = comm->abortFlagDev;
            job->comm = comm;
            ncclIntruQueueEnqueue(asyncJobsMain, (struct ncclAsyncJob *)job);

            struct ncclComm *next = comm->preconnectNext;
            comm->preconnectNext = reinterpret_cast<struct ncclComm *>(0x1);
            comm = next;
        } while (comm != nullptr);
    }

    NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);

    /* Connect channels at runtime if cumem is supported */
    if (groupCommHeadMain != nullptr)
    {
        struct ncclComm *comm = groupCommHeadMain;
        struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs;
        ncclIntruQueueConstruct(&asyncCollJobs);
        do
        {
            bool needConnect = false;
            bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
            memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);

            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
            NCCLCHECKGOTO(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo), ret, fail);

            if (comm->cuMemSupport && needConnect)
            {
                struct ncclPreconnectJob *job;
                NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
                job->base.func = ncclCollPreconnectFunc;
                job->base.undo = nullptr;
                job->base.destructor = free;
                job->base.state = ncclGroupJobRunning;
                job->base.abortFlag = comm->abortFlag;
                job->base.abortFlagDev = comm->abortFlagDev;
                job->comm = comm;
                NCCLCHECKGOTO(ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS), ret, fail);
                memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS);
                ncclIntruQueueEnqueue(&asyncCollJobs, &job->base);
            }
            comm = comm->groupNext;
        } while (comm);

        NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);
        while (!ncclIntruQueueEmpty(&asyncCollJobs))
        {
            struct ncclAsyncJob *job = ncclIntruQueueDequeue(&asyncCollJobs);
            if (job->destructor)
                job->destructor((void *)job);
        }
    }

    if ((!simInfo) && (groupCommHeadMain != nullptr))
    {
        NCCLCHECKGOTO(doLaunches(groupCommHeadMain), ret, fail);
    }

    while (!ncclIntruQueueEmpty(asyncJobsMain))
    {
        struct ncclAsyncJob *job = ncclIntruQueueDequeue(asyncJobsMain);
        if (!job->destroyFlag && job->comm && !job->comm->config.blocking)
            (void)ncclCommSetAsyncError(job->comm, ret);
        if (job->destructor)
            job->destructor((void *)job);
    }

    while (groupCommHeadMain != nullptr)
    {
        struct ncclComm *comm = groupCommHeadMain;
        struct ncclComm *next = comm->groupNext;
        (void)ncclGroupCommLeave(comm);
        if (!comm->config.blocking)
        {
            (void)ncclCommSetAsyncError(comm, ret);
        }
        groupCommHeadMain = next;
    }

    CUDACHECK(cudaSetDevice(savedDev));

exit:
    return ret;
fail:
    groupCleanup(gjob->groupCommHeadPtr, gjob->groupCommPreconnectHeadPtr, gjob->asyncJobsPtr, gjob->groupErrorPtr, gjob->groupBlockingPtr, gjob->abortFlagPtr, ret);
    goto exit;
}
```
* group.cc中
```c
static ncclResult_t doLaunches(struct ncclComm *head)
{
    ncclResult_t result = ncclSuccess;
    struct ncclComm *cliqueComm0 = head->intraComm0;
    struct ncclComm *cliqueHead = head;
    struct ncclComm *cliqueNextHead;
    bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup;
    // This outer loop iterates over cliques of comms which are siblings of the
    // same global entity. We calculate a clique as all comms which have the same
    // `intraComm0` value.
    do
    {
        struct ncclComm *comm = cliqueHead;
        bool capturingYes = false, capturingNo = false;
        do
        {
            (ncclCudaGraphValid(comm->planner.capturingGraph) ? capturingYes : capturingNo) = true;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            /* 在ncclLaunchPrepare里面:
             *      if (planner->nTasksColl == 0 && planner->nTasksP2p != 0)
                    {
                        NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, plan, &budget), result, failure);
                    }
            * 这个scheduleP2pTasksToPlan里面会把p2p的任务加入到planner里面设置了plan->kernelFn
            */
            NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);
            if (useBarrier)
                ncclCommIntraBarrierIn(comm, 1);
            comm = comm->groupNext;
        } while (comm != nullptr && comm->intraComm0 == cliqueComm0);
        cliqueNextHead = comm;

        if (capturingYes && capturingNo)
        {
            // We have entered barriers but are aborting without leaving them. Thus
            // these comms are permanently trashed. We need a good mechanism for
            // tracking and reporting that.
            WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
            result = ncclInvalidUsage;
            goto failure;
        }

        while (true)
        { // Iterate rounds of launches for clique.
            bool moreRounds = false;
            comm = cliqueHead;
            do
            { // Iterate clique members.
                struct ncclComm *next = comm->groupNext;
                if (useBarrier)
                {
                    // Barrier reduction result tells us if this was the final round.
                    moreRounds = 0 != ncclCommIntraBarrierOut(comm);
                }
                else
                {
                    moreRounds |= comm->planner.unlaunchedPlansHead != nullptr;
                }
                if (moreRounds)
                {
                    // Pop next unlaunched kernel
                    struct ncclKernelPlan *plan = comm->planner.unlaunchedPlansHead;
                    if (plan != nullptr)
                    {
                        comm->planner.unlaunchedPlansHead = plan->next;
                        CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
                        NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
                        NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
                    }
                    // Barrier reduction input indicates if we require further rounds.
                    if (useBarrier)
                        ncclCommIntraBarrierIn(comm, comm->planner.unlaunchedPlansHead != nullptr ? 1 : 0);
                    if (plan != nullptr)
                    {
                        NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
                    }
                }
                else
                { // Final round.
                    CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
                    NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
                }
                comm = next;
            } while (comm != cliqueNextHead);
            if (!moreRounds)
                break;
        }
        cliqueHead = cliqueNextHead;
    } while (cliqueHead != nullptr);
failure:
    return result;
}
```

* enqueue.cc中
    * **最后调用cuda执行kernel的时候，fn告知了线程要使用的device.h里面的什么函数**
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
* 注意！！**cuLaunchKernel是异步的**，所以CPU侧的逻辑到此，发了任务就返回了，不用等待CUDA执行完。这也就是nccl-test里面的CPU时间和端到端时间的差别(-C参数是否设置)，CPU的任务到此基本上就结束了，然后开始返回返回返回，端到端时间则是要等待线程同步完成的。这个函数传入的参数，就是后续所有任务的所有参数了。
* 很抽象的一个点，调用sendrecv.h里面函数的地方是一个在make之前不存在的文件，在make的时候generate.py生成了一个nccl/build/obj/device/gensrc/sendrecv.cu，这里面使用了之前代码里声明的一个宏：
```c
// 原代码
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

// 生成的代码
#include "common.h"
#include "sendrecv.h"
DEFINE_ncclDevKernel(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 589)
DEFINE_ncclDevFunc(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)

```


* ncclDevFuncRowToId也是用generate.py生成到host_table.cc里的



