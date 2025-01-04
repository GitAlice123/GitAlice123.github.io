---
title: NCCL代码阅读-05
date: 2024-12-19 09:38:34
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---
## AllReduce操作流程(从ncclLaunchKernel开始)
* 因为我的项目基本上只用allreduce，所以我就重点关注了一下这个操作
* 具体来说我用的是allreduce的u32的sum操作
* ncclLaunchKernel前面的内容和《NCCL代码阅读-01》里面记录的sendrecv操作差不多，就不过多解释了

-----
### 准备工作：launchKernel的前夜
* 这部分本来应该在01里面解释的，但是放在这里，更有助于理解kernel中的workbatch, task等概念
#### taskAppend
* 回顾一下，**taskAppend**函数会在**ncclEnqueueCheck**中被调用，而ncclEnqueueCheck只需要传入一个参数，叫**info**，这个info来自于ncclAllReduce（nccl做AllReduce操作的最上面入口处，nccl-test可以调用的API）
```c
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
```
* 下面我将接着《NCCL代码阅读-nccltest篇》来分析，上面info里面的几个参数如果看过那篇都很熟悉
* 这边单独拿出taskAppend分析一下，这个task会被一直传递到下面的kernel中
* 分析写在注释里了
```c
// 把info转换成一个ncclTaskColl（task），把task加到comm->planner里
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;

  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    // 我们只分析我们的例子，我们的coll是ncclFuncAllReduce
  } else {
    // Empty collectives can be discarded.
    if (info->count == 0) return ncclSuccess;

    // 将主机端的 info->op（操作符）转换成设备端格式，存入 opDev 结构体中，供后续计算使用
    struct ncclDevRedOpFull opDev;
    NCCLCHECK(hostToDevRedOp(&opDev, info->op, info->datatype, comm));

    if (comm->nRanks == 1) {
        //我们的例子中comm->nRanks是2
    } else {
      // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
      // 这里，我们的情况下，是每个进程的一个comm单独成一个group，ncclGroupCommHead就指向各自进程的那一个comm
      // 这里的group，可以查看《NCCL代码阅读-02》
      ncclGroupCommJoin(info->comm);
      // 分配一个ncclTaskColl结构体
      struct ncclTaskColl* t = ncclMemoryPoolAlloc<struct ncclTaskColl>(&comm->memPool_ncclTaskColl, &comm->memPermanent);
      t->func = info->coll;
      t->sendbuff = info->sendbuff;
      t->recvbuff = info->recvbuff;
      t->count = info->count;
      t->root = info->root;
      t->datatype = info->datatype;
      size_t elementSize = ncclTypeSize(t->datatype);
      if (t->func == ncclFuncAllGather || t->func == ncclFuncBroadcast) {
        t->count *= elementSize;
        t->datatype = ncclInt8;
        elementSize = 1;
      }
      // 这边可以点进去看一下，AllReduce操作的ncclFuncTrafficPerByte是2，因为每个字节都要发出一次，收回一次
      t->trafficBytes = t->count*elementSize*ncclFuncTrafficPerByte(t->func, comm->nRanks);
      t->opHost = info->op;
      t->opDev = opDev; // C++ struct assignment
      t->chunkSteps = info->chunkSteps;
      t->sliceSteps = info->sliceSteps;

      //更新当前任务的总数 nTasksColl。
      //将新任务按流量大小插入任务队列中，ncclTaskCollSorterInsert 根据任务的流量字节数排序。
      planner->nTasksColl += 1;
      ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
    }
  }

  if (info->stream != planner->streamRecent || planner->streams == nullptr) {
    planner->streamRecent = info->stream;
    struct ncclCudaStreamList* l = planner->streams;
    while (true) {
      if (l == nullptr) { // Got to the end, this must be a new stream.
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream));
        if (planner->streams != nullptr && !ncclCudaGraphSame(planner->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        planner->capturingGraph = graph; // C++ struct assignment
        // Add stream to list
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;
        l->next = planner->streams;
        planner->streams = l;
        break;
      }
      if (l->stream == info->stream)
        break; // Already seen stream.
      l = l->next;
    }
  }
  return ncclSuccess;
}

```
* 我们在这里暂停一下，总结一下。appendTask之后，我们的一个task（AllReduce，数据元素个数为count个）就被插入comm的planner里面了，所有的命令信息都在这里面。
* 从这里往后，都是以group为单位进行操作，我们这边只要记得ncclGroupCommHead就是指向group中第一个comm的指针即可，在我们的场景下，就指向每个进程那唯一的一个comm

#### ncclGroupEndInternal
```c
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) {
  ncclResult_t ret = ncclSuccess;
  ncclSimInfo_t internalSimInfo = NCCL_SIM_INFO_INITIALIZER;
  ncclSimInfo_t* internalSimInfoPtr = NULL;
  size_t realSize = 0;

  internalSimInfo.magic = 0;

  if (ncclGroupDepth == 0) {
    WARN("ncclGroupEnd: not in a group call.");
    ret = ncclInvalidUsage;
    goto exit;
  }

  if ((--ncclGroupDepth) > 0) goto exit;

  if ((ret = ncclGroupError) != ncclSuccess) goto fail;

  if (simInfo) {
    // 不用管，用不到
  }

  if (ncclGroupCommHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    // 记得上面的，ncclGroupCommHead就是comm
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
    if (ncclGroupBlocking == 0) {
      /* nonblocking group */
      // 但我们是blocking group，暂时不看这部分
    } else {
      /* blocking group */
      NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base, internalSimInfoPtr), ret, fail);
      if (simInfo) memcpy((void*)simInfo, (void*)internalSimInfoPtr, realSize);
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
#### groupLaunch
##### groupLaunch
```c
/**
传入的参数中，job_指向这一个group里面所有的异步任务的链表
 */
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL) {
  int savedDev;
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  struct ncclComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  struct ncclComm *groupCommPreconnectHeadMain = *gjob->groupCommPreconnectHeadPtr;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = gjob->asyncJobsPtr;

  bool *groupAbortFlag = gjob->abortFlagPtr;

  CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);

  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    //  不走这一段
  }

  // 没有预连接任务的话，这个函数也是直接返回的
  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);

  /* Connect channels at runtime if cumem is supported */
  if (groupCommHeadMain != nullptr) {
    struct ncclComm* comm = groupCommHeadMain;
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs;
    ncclIntruQueueConstruct(&asyncCollJobs);
    do {
      bool needConnect = false;
      bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
      memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);

      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
      NCCLCHECKGOTO(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo), ret, fail);

      if (comm->cuMemSupport && needConnect) {
        struct ncclPreconnectJob* job;
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
    while (!ncclIntruQueueEmpty(&asyncCollJobs)) {
      struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs);
      if (job->destructor) job->destructor((void*)job);
    }
  }

  if ((!simInfo) && (groupCommHeadMain != nullptr)) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain), ret, fail);
  }

  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    if (!job->destroyFlag && job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, ret);
    if (job->destructor) job->destructor((void*)job);
  }

  while (groupCommHeadMain != nullptr) {
    struct ncclComm* comm = groupCommHeadMain;
    struct ncclComm* next = comm->groupNext;
    (void) ncclGroupCommLeave(comm);
    if (!comm->config.blocking) {
      (void) ncclCommSetAsyncError(comm, ret);
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

* 接下来，就进入了doLaunches，带着的参数说白了就是comm
* 看groupLaunch之前，要先看一下ncclAsyncJob这个结构体
* 以及，ncclGroupJob这个结构体是继承自ncclAsyncJob的
##### ncclAsyncJob
```c
struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  pthread_t thread;
  ncclResult_t result;
  ncclResult_t(*func)(struct ncclAsyncJob*);
  void(*undo)(struct ncclAsyncJob*);
  void(*destructor)(void*);
  ncclGroupJobState_t state;
  uint32_t* abortFlag; /* point to comm abortFlag */
  uint32_t* abortFlagDev; /* point to comm abortFlagDev */
  uint32_t* childAbortFlag; /* point to child abortFlag */
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */
  // 应该还记得，我们的实际任务被插入comm的planner里面了
  ncclComm_t comm;
  int destroyFlag;
};
```
##### ncclPrepareTasks
* 还要再看一下ncclPrepareTasks函数，在groupLaunch里面被调用的，这里面:
    * 确定了nTasksPerChannel
    * 一系列操作（这边我没具体分析，后面如果有需要我会回来补充），将tasks加入了planner->collTaskQueue中
    * 下面出现了两个结构：ncclDevWorkColl devWork和ncclWorkList* workNode
    * devWork 是一个包含集体通信任务的详细信息的结构体（ncclDevWorkColl）。
    * workNode 是一个 ncclWorkList 结构体，它包含一个类型字段（workType）和一个大小字段（size），以及存储任务数据的区域
    * workNode 包含了 devWork 作为它的数据部分
    * 把task转换成了ncclDevWorkColl，然后加入workNode，最后加入了队列：
    ```c
    ncclIntruQueueEnqueue(&planner->collWorkQueue, workNode)
    ```
    * 在我的场景中，两个comm只有第一次要初始化algo channel，所以needConnect各自只有一次是true

#### doLaunches
* 主要功能就是两层循环，外层遍历每个comm组，内层处理组里面每个comm
* 对每个comm做初始化：ncclLaunchPrepare
* 从 comm->planner.unlaunchedPlansHead 中获取下一个待执行的内核计划（ncclKernelPlan）。
* 在启动内核前，执行ncclLaunchKernelBefore_NoUncapturedCuda
* 执行ncclLaunchKernel来启动内核
* 启动内核之后，执行ncclLaunchKernelAfter_NoCuda。

##### ncclLaunchPrepare
* 又出来一个新的结构：ncclKernelPlan.......（怎么这么多结构啊
* 分配了一个plan结构
* 总之我们是集合操作，所以进入了scheduleCollTasksToPlan(comm, plan, &budget)，下面有这个函数的基本内容。这个函数主要干了这些事：
    * 划分了channel和chunk
    * 把task从comm的planner队列中取出，加到plan->collTaskQueue里
    * 在workNode后面带着的devWork里面存入划分出的chunk、channel等信息，最后把workNode从comm的planner队列中取出，加到plan->WorkQueue里
    * 在plan里面存入channelMask等信息
* 调用finishPlan函数（下面有代码），这个函数主要做的是：
    * 在plan里面分配一个kernelArgs结构，然后设置其comm, channelMask和workStorageType
    * 下面把所有channel上的workBatchQueue里面的所有workBatch都按照一定顺序挂到kernelArgs后面去：
    ```c
    struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
    ```
* 然后把这个plan加入comm的planQueue
* 设置了流之间的依赖关系


###### scheduleCollTasksToPlan
```c
static ncclResult_t scheduleCollTasksToPlan(
    struct ncclComm *comm, struct ncclKernelPlan *plan, struct ncclKernelPlanBudget *budget)
{
    struct ncclKernelPlanner *planner = &comm->planner;
    // Estimate number of tasks that will fit in this plan.
    int nPlanColls = 0;
    size_t trafficBytes[2 * 2] = {0, 0, 0, 0};                            // [collnet][nvls]
    int nChannels[2 * 2] = {0, 0, 0, 0};                                  // [collnet][nvls]
    int const nMaxChannels[2 * 2] = {comm->nChannels, comm->nvlsChannels, // [collnet][nvls]
                                     comm->nChannels, comm->nvlsChannels};
    constexpr size_t MinTrafficPerChannel = 16 << 10; // 16K traffic as minimal
    do
    {
        size_t workBytes = 0;
        struct ncclTaskColl *task = ncclIntruQueueHead(&planner->collTaskQueue);
        struct ncclWorkList *workNode = ncclIntruQueueHead(&planner->collWorkQueue);
        while (task != nullptr)
        {
            int nBatches = divUp(nPlanColls, 4); // Rough guess: 4 colls per batch.
            if (!testBudget(budget, nBatches, workBytes + workNode->size))
                goto plan_full;

            nPlanColls += 1;
            workBytes += workNode->size;
            int kind = 2 * task->isCollnet + task->isNvls;
            trafficBytes[kind] += std::max(MinTrafficPerChannel, task->trafficBytes);//2M
            nChannels[kind] += task->nMaxChannels;
            nChannels[kind] = std::min(nChannels[kind], nMaxChannels[kind]);//2
            task = task->next;
            workNode = workNode->next;
        }
    plan_full:;
    } while (0);

    int kindPrev = -1;
    size_t trafficPerChannel = 0;
    int channelId = 0;
    size_t currentTraffic = 0;
    while (nPlanColls != 0 && !ncclIntruQueueEmpty(&planner->collTaskQueue))
    {
        struct ncclTaskColl *task = ncclIntruQueueHead(&planner->collTaskQueue);
        struct ncclWorkList *workNode = ncclIntruQueueHead(&planner->collWorkQueue);
        struct ncclDevWorkColl *devWork = (struct ncclDevWorkColl *)(workNode + 1);
        size_t elementSize = ncclTypeSize(task->datatype);

        int kind = 2 * task->isCollnet + task->isNvls;
        if (kind != kindPrev)
        {
            trafficPerChannel = std::max<size_t>(MinTrafficPerChannel, trafficBytes[kind] / nChannels[kind]);//1M
            kindPrev = kind;
            channelId = 0;
            currentTraffic = 0;
        }

        if (task->isCollnet)
        {
            //我们的实验里不是
        }
        else
        { // not task->isCollnet
            int trafficPerByte = ncclFuncTrafficPerByte(task->func, comm->nRanks);//2
            size_t cellSize = divUp(divUp(MinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;//8192
            int elementsPerCell = cellSize / elementSize;//2048
            size_t cells = divUp(task->count * elementSize, cellSize);//128
            size_t trafficPerElement = elementSize * trafficPerByte;//8
            size_t trafficPerCell = cellSize * trafficPerByte;//16384
            size_t cellsPerChannel = std::min(cells, divUp(trafficPerChannel, trafficPerCell));//64
            size_t cellsLo;//64
            if (channelId + 1 == nMaxChannels[kind])
            { // On last channel everything goes to "lo"
                cellsLo = cells;
            }
            else
            {
                cellsLo = std::min(cells, divUp((trafficPerChannel - currentTraffic), trafficPerCell));
            }
            int nMidChannels = (cells - cellsLo) / cellsPerChannel;//1
            size_t cellsHi = (cells - cellsLo) % cellsPerChannel;//0
            int nChannels = (cellsLo != 0 ? 1 : 0) + nMidChannels + (cellsHi != 0 ? 1 : 0);//2
            if (nMaxChannels[kind] < channelId + nChannels)
            { // Overflowed available channels
                nMidChannels = nMaxChannels[kind] - channelId - 2;
                cellsPerChannel = (cells - cellsLo) / (nMidChannels + 1);
                cellsHi = cellsPerChannel + (cells - cellsLo) % (nMidChannels + 1);
            }
            if (cellsHi == 0 && nMidChannels != 0)
            {
                cellsHi = cellsPerChannel;
                nMidChannels -= 1;
            }
            if (cellsLo == 0)
            { // Least channel skipped. Make the next channel the new least.
                channelId += 1;
                if (nMidChannels == 0)
                {
                    cellsLo = cellsHi;
                    cellsHi = 0;
                }
                else
                {
                    cellsLo = cellsPerChannel;
                    nMidChannels -= 1;
                }
            }
            size_t countMid = nMidChannels != 0 ? cellsPerChannel * elementsPerCell : 0;
            size_t countLo = cellsLo * elementsPerCell;
            size_t countHi = cellsHi * elementsPerCell;
            //countLo = 131072, countMid = 0, countHi = 131072
            (countHi != 0 ? countHi : countLo) -= cells * elementsPerCell - task->count;

            nChannels = (countLo != 0 ? 1 : 0) + nMidChannels + (cellsHi != 0 ? 1 : 0);//2
            // Ensure room for worst case of one new batch per channel
            if (!testBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size))
            {
                return ncclSuccess;
            }

            devWork->channelLo = channelId;//0
            devWork->channelHi = channelId + nChannels - 1;//1
            devWork->cbd.countLo = countLo;
            devWork->cbd.countMid = countMid;
            devWork->cbd.countHi = countHi;

            // calcCollChunking() uses global bytes instead of traffic which differs
            // in that allreduce isn't multiplied by 2.
            size_t globalBytesPerElement = elementSize * ncclFuncMaxSendRecvCount(task->func, comm->nRanks, 1);//4
            struct ncclProxyOp proxyOpLo, proxyOpMid, proxyOpHi;

            uint32_t chunkSize, directFlags = 0;
            size_t grainSize = ncclProtoGrainSize(task->protocol);//1920
            if (countLo != 0)
            {
                NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement * countLo, &chunkSize, &directFlags, &proxyOpLo));
                devWork->cbd.chunkGrainsLo = chunkSize / grainSize;
            }
            if (countHi != 0)
            {
                NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement * countHi, &chunkSize, &directFlags, &proxyOpHi));
                devWork->cbd.chunkGrainsHi = chunkSize / grainSize;
            }
            if (nMidChannels != 0)
            {
                NCCLCHECK(calcCollChunking(comm, task, /*nChannels=*/1, globalBytesPerElement * countMid, &chunkSize, &directFlags, &proxyOpMid));
                devWork->cbd.chunkGrainsMid = chunkSize / grainSize;
            }
            devWork->direct = directFlags;

            // Update the current channel and vacant traffic budget.
            if (countHi != 0)
            {
                channelId += nChannels - 1;
                currentTraffic = cellsHi * elementsPerCell * trafficPerElement;
            }
            else if (nMidChannels != 0)
            {
                channelId += nChannels;
                currentTraffic = 0;
            }
            else
            {
                currentTraffic += cellsLo * elementsPerCell * trafficPerElement;
            }

            if (currentTraffic >= trafficPerChannel && channelId + 1 != nMaxChannels[kind])
            {
                channelId += 1;
                currentTraffic = 0;
            }

            // 下面这个循环里面本来有一大堆关于proxy的，但是我们采用的方法用不到proxy，过
            for (int c = devWork->channelLo; c <= (int)devWork->channelHi; c++)
            {
                // 每个channel都有一个workbatchqueue，每个workbatch的内容很少，只指定了操作类型（allreduce+ring）这种
                addWorkBatchToPlan(comm, plan, c, workNode->workType, task->devFuncId, plan->workBytes);
            }
        }

        // 确定channelMask
        plan->channelMask |= (2ull << devWork->channelHi) - (1ull << devWork->channelLo);
        // 确定一个线程块的线程数
        plan->threadPerBlock = std::max(plan->threadPerBlock, task->nWarps * WARP_SIZE);
        if (!plan->kernelSpecialized)
        {
            plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
            plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];
        }

        if (comm->rank == 0)
        {
            if (task->isCollnet)
            {
                // 过
            }
            else
            {
                TRACE(NCCL_COLL, "Collective %s(%s, %s, %s, %s) count=%ld devFuncId=%d channel{Lo..Hi}={%d..%d} count{Lo,Mid,Hi}={%ld,%ld,%ld} chunkBytes{Lo,Mid,Hi}={%d,%d,%d}",
                      ncclFuncToString(task->func), ncclDevRedOpToString(task->opDev.op),
                      ncclDatatypeToString(task->datatype), ncclAlgoToString(task->algorithm),
                      ncclProtoToString(task->protocol),
                      (long)task->count, task->devFuncId, devWork->channelLo, devWork->channelHi,
                      (long)devWork->cbd.countLo, (long)devWork->cbd.countMid, (long)devWork->cbd.countHi,
                      int(devWork->cbd.chunkGrainsLo * ncclProtoGrainSize(task->protocol)),
                      int(devWork->cbd.chunkGrainsMid * ncclProtoGrainSize(task->protocol)),
                      int(devWork->cbd.chunkGrainsHi * ncclProtoGrainSize(task->protocol)));
            }
        }

        for (int i = 0; i < task->nCleanupQueueElts; i++)
        {
            ncclIntruQueueEnqueue(&plan->cleanupQueue, ncclIntruQueueDequeue(&planner->collCleanupQueue));
        }
        // 从planner->collTaskQueue排出一个task元素
        ncclIntruQueueDequeue(&planner->collTaskQueue);
        // 从planner->collWorkQueue排出一个workNode
        ncclIntruQueueDequeue(&planner->collWorkQueue);
        nPlanColls -= 1;
        planner->nTasksColl -= 1;
        // 把排出的task和workNode加入plan的collTaskQueue和workQueue
        ncclIntruQueueEnqueue(&plan->collTaskQueue, task);
        ncclIntruQueueEnqueue(&plan->workQueue, workNode);
        plan->workBytes += workNode->size;
    }
    return ncclSuccess;
}

```

###### finishPlan
```c
static void finishPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
    // 这里面设置了kernelArgs，这是真的最后要给内核的参数
  ncclKernelPlanner::WipPlan::Channel* wipChannels = comm->planner.wipPlan.channels;
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);

  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MIN_NTHREADS);

  // If we can fit everything into the kernel args we do so.
  if (sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes) {
    // 我们的例子里，能放得下
    plan->workStorageType = ncclDevWorkStorageTypeArgs;
  }
  plan->kernelArgsSize = sizeof(struct ncclDevKernelArgs) + batchBytes;
  plan->kernelArgsSize += (plan->workStorageType == ncclDevWorkStorageTypeArgs) ? workBytes : 0;
  plan->kernelArgsSize = alignUp(plan->kernelArgsSize, 16);
  // 分配出一个kernelArgs，然后设置comm,channelMask和workStorageType
  plan->kernelArgs = (struct ncclDevKernelArgs*)ncclMemoryStackAlloc(&comm->memScoped, plan->kernelArgsSize, /*align=*/16);
  plan->kernelArgs->comm = comm->devComm;
  plan->kernelArgs->channelMask = plan->channelMask;
  plan->kernelArgs->workStorageType = plan->workStorageType;

  // Put batches into the kernel arguments. The first batch for each channel
  // must be located at batchZero[blockIdx.x]. To achieve this we round robin
  // over the channels in ascending order until they're exhausted.
  uint64_t hasBatchMask = plan->channelMask;
  struct ncclDevWorkBatch* batchPrev[MAXCHANNELS] = {}; // {0...}
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  int batchIx = 0;
  while (hasBatchMask != 0) {
    uint64_t tmpMask = hasBatchMask; // channels with a batch for this round.
    do {
      int c = popFirstOneBit(&tmpMask);
      if (!ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        struct ncclWorkBatchList* batchNode = ncclIntruQueueDequeue(&wipChannels[c].workBatchQueue);
        if (batchPrev[c] != nullptr) {
          batchPrev[c]->nextJump = int(&batchZero[batchIx] - batchPrev[c]);
        }
        batchPrev[c] = &batchZero[batchIx];
        batchZero[batchIx++] = batchNode->batch;
      }
      if (ncclIntruQueueEmpty(&wipChannels[c].workBatchQueue)) {
        hasBatchMask ^= 1ull<<c;
      }
    } while (tmpMask != 0);
  }

  // Merge-sort per-channel proxy-op lists by opCount when merging them into plan->proxyOpQueue
  // Phase 1: scan first op of each channel, store opCount in headIds[c].
  uint64_t headIds[MAXCHANNELS];
  int nHeads = 0;
  int channelUbound = 0;
  for (int c=0; c < MAXCHANNELS; c++) {
    struct ncclProxyOp* op = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = op ? op->opCount : uint64_t(-1);
    if (op) nHeads += 1;
    if (op) plan->hasProxyOps = true;
    if (op) channelUbound = c+1;
  }
  // Phase 2: Dequeue from planner->channels[c], enqueue in merged order to plan
  while (nHeads != 0) {
    int c = -1;
    uint64_t minId = uint64_t(-1);
    // Find channel with least proxy-op id. We store the heads[c]->opCount in
    // headIds[c] to remove indirect loads from this loop.
    for (int c1=0; c1 < channelUbound; c1++) {
      uint64_t id = headIds[c1];
      id = (id>>1 | id<<63); // Move tag bit to order collectives before p2p's
      if (id < minId) { c = c1; minId = id; }
    }
    struct ncclProxyOp* op = ncclIntruQueueDequeue(&wipChannels[c].proxyOpQueue);
    struct ncclProxyOp* opNext = ncclIntruQueueHead(&wipChannels[c].proxyOpQueue);
    headIds[c] = opNext ? opNext->opCount : uint64_t(-1);
    nHeads -= opNext ? 0 : 1;
    ncclIntruQueueEnqueue(&plan->proxyOpQueue, op);
  }
}
```

##### ncclLaunchKernelBefore_NoUncapturedCuda
* 调用函数uploadWork
```c
static ncclResult_t uploadWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  size_t workBytes = plan->workBytes;
  size_t batchBytes = plan->nWorkBatches*sizeof(struct ncclDevWorkBatch);
  void* fifoBufHost;
  uint32_t fifoCursor, fifoMask;

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeArgs:
    // 我们是这个情况
    plan->kernelArgs->workBuf = nullptr;
    fifoBufHost = (void*)plan->kernelArgs;
    fifoCursor = sizeof(ncclDevKernelArgs) + batchBytes;
    fifoMask = ~0u;
    break;
  case ncclDevWorkStorageTypeFifo:
    // fifoBufHost = comm->workFifoBuf;
    // fifoCursor = comm->workFifoProduced;
    // fifoMask = comm->workFifoBytes-1;
    // waitWorkFifoAvailable(comm, fifoCursor + workBytes);
    // plan->kernelArgs->workBuf = comm->workFifoBufDev;
    // break;
  case ncclDevWorkStorageTypePersistent:
    // static_assert(16 <= alignof(max_align_t), "We rely on 16-byte alignment.");
    // fifoBufHost = malloc(workBytes);
    // fifoCursor = 0;
    // fifoMask = ~0u;
    // break;
  default:
    return ncclInternalError;
  }
  plan->kernelArgs->workMask = fifoMask;

  // Batches were placed after kernelArgs by finishPlan(). Only thing left to
  // do is translate the work offset from zero based (in plan) to:
  //  ncclDevWorkStorageTypeArgs: offset from beginning of kernel args
  //  ncclDevWorkStorageTypeFifo: offset from base of fifo
  //  ncclDevWorkStorageTypePersistent: no translation since our dedicated buffer will also begin at zero.
  struct ncclDevWorkBatch* batchZero = (struct ncclDevWorkBatch*)(plan->kernelArgs+1);
  for (int b=0; b < plan->nWorkBatches; b++) {
    batchZero[b].offsetBase += fifoCursor;
  }

  // 这边的FIFO队列是主机端的，直接指向同样放在主机内存的plan->kernelArgs，后面在启用内核函数的时候，指向kernelArgs的指针会被作为参数传到内核函数里，然后kernelArgs的内容会被复制到GPU的共享内存里
  // Write the channel-shared work structs.
  struct ncclWorkList* workNode = ncclIntruQueueHead(&plan->workQueue);
  while (workNode != nullptr) {
    char* dst = (char*)fifoBufHost;
    char* src = (char*)(workNode+1);
    for (int n = workNode->size; n != 0; n -= 16) {
      memcpy(
        __builtin_assume_aligned(dst + (fifoCursor & fifoMask), 16),
        __builtin_assume_aligned(src, 16),
        16
      );
      fifoCursor += 16;
      src += 16;
    }
    workNode = workNode->next;
  }

  switch (plan->workStorageType) {
  case ncclDevWorkStorageTypeFifo:
    // comm->workFifoProduced = fifoCursor;
    // if (comm->workFifoBufGdrHandle != nullptr) wc_store_fence();
    // break;
  case ncclDevWorkStorageTypePersistent:
    // { ncclResult_t result = ncclSuccess;
    //   cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    //   void* fifoBufDev = nullptr;
    //   CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));

    //   // Acquire deviceStream to gain access to deviceStream.cudaStream. Since the
    //   // user's graph will be launched later, and it also acquires the deviceStream,
    //   // it will observe this upload.
    //   NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream), result, finish_scope);

    //   CUDACHECKGOTO(cudaMallocAsync(&fifoBufDev, workBytes, comm->memPool, comm->sharedRes->deviceStream.cudaStream), result, finish_scope);
    //   plan->workBufPersistent = fifoBufDev;
    //   plan->kernelArgs->workBuf = fifoBufDev;

    //   CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, comm->sharedRes->deviceStream.cudaStream), result, finish_scope);
    //   cudaEvent_t memcpyDone;
    //   CUDACHECKGOTO(cudaEventCreateWithFlags(&memcpyDone, cudaEventDisableTiming), result, finish_scope);
    //   CUDACHECKGOTO(cudaEventRecord(memcpyDone, comm->sharedRes->deviceStream.cudaStream), result, finish_scope);

    //   struct uploadWork_cleanup_t* cleanup;
    //   NCCLCHECK(ncclCalloc(&cleanup, 1));
    //   cleanup->base.fn = uploadWork_cleanup_fn;
    //   cleanup->base.event = memcpyDone;
    //   cleanup->hostBuf = fifoBufHost;
    //   ncclIntruQueueEnqueue(&comm->eventCallbackQueue, &cleanup->base);

    //   NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream), result, finish_scope);
    //   NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm), result, finish_scope);

    // finish_scope:
    //   CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    //   if (result != ncclSuccess) return result;
    // } break;
  default: break;
  }
  return ncclSuccess;
}

```
##### ncclLaunchKernel
```c
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  // 开启channel的数量，在我们的例子里是2个
  int nChannels = countOneBits(plan->channelMask);
  void* sym = plan->kernelFn;
  // 可以看到，其实是一个一维的grid，有多少个channel就有几个block
  dim3 grid = {(unsigned)nChannels, 1, 1};
  // 我们的例子里threadPerBlock是640
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  int smem = ncclShmemDynamicSize(comm->cudaArch);
  cudaStream_t launchStream = planner->streams->stream;
  void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize,
    CU_LAUNCH_PARAM_END
  };

  CUfunction fn;
  CUDACHECK(cudaGetFuncBySymbol(&fn, sym));

  #if CUDART_VERSION >= 11080
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion >= 11080) {
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
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain) ncclParamMemSyncDomain();
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

    //CUDACHECK(cudaLaunchKernelExC(&launchConfig, fnAddr, args));
    CUCHECK(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra));
    return ncclSuccess;
  }
  #endif
  // Standard kernel launch
  CUCHECK(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra));
  //CUDACHECK(cudaLaunchKernel(fnAddr, grid, block, args, smem, launchStream));
  return ncclSuccess;
}
```
* 调用了cuLaunchKernel(Ex)，交给GPU异步处理，CPU可以返回了

### 生成文件里的内容
```c
#include "common.h"
#include "all_reduce.h"
DEFINE_ncclDevKernel(AllReduce_Sum_u32_RING_LL, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_RING, NCCL_PROTO_LL, 230)
DEFINE_ncclDevKernel(AllReduce_Sum_u32_TREE_LL, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_TREE, NCCL_PROTO_LL, 233)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_COLLNET_CHAIN_SIMPLE, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_COLLNET_DIRECT_SIMPLE, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE)
#if CUDART_VERSION >= 12010 && __CUDA_ARCH__ >= 900
DEFINE_ncclDevFunc(AllReduce_Sum_u32_NVLS_SIMPLE, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE)
#endif
#if CUDART_VERSION >= 12010 && __CUDA_ARCH__ >= 900
DEFINE_ncclDevFunc(AllReduce_Sum_u32_NVLS_TREE_SIMPLE, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE)
#endif
DEFINE_ncclDevFunc(AllReduce_Sum_u32_RING_LL, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_RING, NCCL_PROTO_LL)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_RING_LL128, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_RING, NCCL_PROTO_LL128)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_RING_SIMPLE, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_TREE_LL, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_TREE, NCCL_PROTO_LL)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_TREE_LL128, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_TREE, NCCL_PROTO_LL128)
DEFINE_ncclDevFunc(AllReduce_Sum_u32_TREE_SIMPLE, ncclFuncAllReduce, FuncSum, uint32_t, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE)

```
* 按照《NCCL代码阅读-01》说的那样，接下来CUDA就开始查表，接着按照生成文件里的代码，进入了ncclKernelMain
* 注意是从cuLaunchKernel或者cuLaunchKernelEx进入的，在进入的时候，指定了grid和block的维度，后面就会分配多少线程块给他
* 以及，进入的时候，具体的操作内容都是放在extra里面传入的
```c
// 下面是参数实例化之后的调用情况
ncclKernelMain<240, RunWorkBatch<ncclFuncAllReduce, uint64_t, FuncSum<uint64_t>, 
																NCCL_ALGO_RING, NCCL_PROTO_LL>>(&args4K.args)
```
### ncclKernelMain
* 解释都写在注释里面了
```c
template<int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
	//////////////////////////////////////////////////////////////////
	// SpecializedFnId=240
	// SpecializedRunWorkBatch=RunWorkBatch<ncclFuncAllReduce, uint64_t, FuncSum<uint64_t>, 
	//                                      NCCL_ALGO_RING, NCCL_PROTO_LL>>(&args4K.args)
	//////////////////////////////////////////////////////////////////

	// tid是该线程在线程块中的索引，关于CUDA的编程模型见《CUDA编程模型》章节
	int tid = threadIdx.x;
    // tn是该线程块一共有多少线程
	int tn = blockDim.x;

    // 把一些只读的kernel参数放到共享内存里，要是不显示放置的话，编译器会把这些参数放到线程自己的栈里面，很占地方
    // 这里是召集一群thread，把参数拆一拆，每个进程搬一点，把他们搬到共享内存里，后面__syncthreads()就是搬完了
	if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
		((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
	}

    // args->channelMask是一个掩码，比如有64个通道，那就有64位掩码，第x位为1表示第x个通道启用了
    // (1ull<<tid)表示把1左移tid位，这里先把tid看成一个数字即可，所以(1ull<<tid)就是只有第tid位为1
    // 这个if条件表示，满足tid小于MAXCHANNELS（其实就是32）并且第tid位的通道是启用的，才进入if
    // ((1ull<<tid)-1)表示让小于tid位的位全部置1，其余位都是0
    // 比如原来(1ull<<tid)是0b0100，tid是2，那-1之后就是0b0011
    // args->channelMask & ((1ull<<tid)-1)就是只保留小于tid的启用通道
    // __popcll(x)会统计x的二进制下有几个1
    // 所以__popcll(args->channelMask & ((1ull<<tid)-1))统计了小于tid的通道号的通道有几个启用的
    // 最后如果n=线程块号，那就让这个线程块负责tid号channel
    // 其实这种分配方式就是为了做到动态分配线程块所负责的通道，同一个线程块只有一个线程的tid号会被选中作为该线程块的通道号，因为比如2号前面启用的个数一定和1号前面启用的个数不同（在1号启用的条件下），所以不可能同时让n=块号
	if (tid < MAXCHANNELS && (args->channelMask & (1ull<<tid))) {
		int n = __popcll(args->channelMask & ((1ull<<tid)-1));
		if (blockIdx.x == n) ncclShmem.channelId = tid;
	}
	__syncthreads(); // publish ncclShmem.{args, channelId}
	/* set abort flag to 0 */
	if (tid == 0) ncclShmem.aborted = 0;

    // 用前两个warp来搬运comm和channel的控制信息，其余的warp用来搬运work batch
	switch (tid/WARP_SIZE) {
	case 0:
		{ void* dst = &ncclShmem.comm;
			void* src = ncclShmem.args.comm;
			int bytes = sizeof(ncclDevComm);
			static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
			copyToShmem16(tid, dst, src, bytes);
		} break;
	case 1:
		{ // Get address of channel without incurring indirect load from ncclDevComm::channels
			void* dst = &ncclShmem.channel;
			void* src = &((ncclDevCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];
			int bytes = sizeof(ncclDevChannel);
			static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
			copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
		} break;
	default:
		{ int subtid = tid - 2*WARP_SIZE;
			int subtn = tn - 2*WARP_SIZE;
			// Coverity reports a possible thread divergence due to not all threads participating in the collective.
			// However, the code ensures that the participation is on a per-warp basis.
			// coverity[device_thread_diverged:FALSE]
			loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
		} break;
	}
	__syncthreads(); // publish ncclShmem

    // 我们的例子里面workStorageType是ncclDevWorkStorageTypeArgs
	if (tid == 0 && ncclShmem.args.workStorageType == ncclDevWorkStorageTypeFifo) {
		// ncclShmem.workConsumed written by loadWorkBatchToShmem before __syncthreads()
		ncclShmem.comm.workConsumed[ncclShmem.channelId] = ncclShmem.workConsumed;
	}

	while (true) {
		if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
            // 实际执行，RunWorkBatch<ncclFuncAllReduce, uint64_t, FuncSum<uint64_t>, NCCL_ALGO_RING, NCCL_PROTO_LL>>(&args4K.args).run()
			SpecializedRunWorkBatch().run();
		} else {
			ncclDevFuncTable[ncclShmem.funcId]();
		}

		if (ncclShmem.nextBatchIx == -1) break;
		int batchIx = ncclShmem.nextBatchIx;
		__syncthreads();
		loadWorkBatchToShmem(tid, tn, args, batchIx);

		// Check whether the last operation was aborted and make sure all threads exit
		bool aborted = false;
		if (tid == 0) aborted = *ncclShmem.comm.abortFlag;
		aborted = barrier_red_or_aligned(aborted, 0); // publish ncclShmem.work
		if (tid == 0 && ncclShmem.args.workStorageType == ncclDevWorkStorageTypeFifo) {
			// ncclShmem.workConsumed written by loadWorkBatchToShmem before barrier_red_or()
			ncclShmem.comm.workConsumed[ncclShmem.channelId] = ncclShmem.workConsumed;
		}
		if (aborted) break;
	}
}

```
* 可以看到，核心执行函数就是SpecializedRunWorkBatch().run()（上下两个一个意思，去01里提到的表里找一下即可）

### RunWorkBatch
```c
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
/////////////////////////////////////////////////////////////////////////////////////////
// Fn = ncclFuncAllReduce
// T = uint64_t
// RedOp = FuncSum<uint64_t>
// Algo = NCCL_ALGO_RING
// Proto = NCCL_PROTO_LL
/////////////////////////////////////////////////////////////////////////////////////////
	__device__ __forceinline__ void run() {
        // 线程号
		int tid = threadIdx.x;
        // block中线程总数
		int tn = blockDim.x;
		if (RedOpArg<RedOp>::ArgUsed) {
			int nWorks = ncclShmem.nWorks;
			for (int w=tid; w < nWorks; w += tn) {
				struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
				if (work->redOpArgIsPtr) {
					work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
				}
			}
			__syncthreads();
		}

		#pragma unroll 1
		for (int w=0; w < ncclShmem.nWorks; w++) {
			struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
			if (w != 0) {
				struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w-1)*ncclShmem.workSize);
				if (work->nWarps != workPrev->nWarps) __syncthreads();
			}
			int subtn = work->nWarps*WARP_SIZE;
			// Coverity reports a possible thread divergence due to not all threads participating in the collective.
			// However, the code ensures that the participation is on a per-warp basis.
			// coverity[device_thread_diverged:FALSE]
			if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
		}
	}
};
```
**预知后事如何，参见NCCL代码阅读-06**