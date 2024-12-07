---
title: NCCL代码中常用的函数和宏定义
date: 2024-11-29 10:05:31
tags:
- NCCL
- 代码阅读
categories:
- NCCL
---

# NCCLCHECK
`NCCLCHECK` 是一个宏，用于简化 NCCL 函数调用后的错误检查。在 NCCL 和许多 C/C++ 编程环境中，错误处理通常是一个关键部分，而通过宏封装可以使代码更加简洁和易于维护。

---

## **NCCLCHECK 的典型定义**
在 NCCL 的代码中，`NCCLCHECK` 通常是定义为类似下面的宏：

```c
#define NCCLCHECK(call) do { \
  ncclResult_t result = call; \
  if (result != ncclSuccess) { \
    printf("NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(result)); \
    return result; \
  } \
} while(0)
```

## **功能**
1. **执行函数调用并捕获返回值**  
   `call` 是需要执行的 NCCL 函数，比如 `ncclInit()` 或 `PtrCheck(out, "GetUniqueId", "out")`。这些函数通常返回一个类型为 `ncclResult_t` 的结果，用于指示是否成功。

2. **检查返回值是否成功**  
   如果 `call` 返回的值不是 `ncclSuccess`，则表示调用失败。

3. **打印调试信息**  
   如果失败，宏会打印文件名、行号以及错误字符串。`ncclGetErrorString` 是 NCCL 提供的函数，可以将错误码转换为可读的错误消息。

4. **中止当前流程**  
   如果函数调用失败，`NCCLCHECK` 通常会返回错误码，退出当前函数。

---

## **使用示例**
在代码中，`NCCLCHECK` 的作用是捕获和处理 NCCL 函数的错误。例如：
```c
NCCLCHECK(ncclInit());
```

等价于：
```c
{
  ncclResult_t result = ncclInit();
  if (result != ncclSuccess) {
    printf("NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(result));
    return result;
  }
}
```

---

## **代码中的用途**
在 `ncclGetUniqueId` 函数中，`NCCLCHECK` 用来确保：
1. **NCCL 初始化成功：**
   ```c
   NCCLCHECK(ncclInit());
   ```
   如果 `ncclInit()` 返回错误码，函数将立即返回错误。

2. **指针有效性检查：**
   ```c
   NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
   ```
   如果 `out` 指针无效或检查失败，函数会打印错误信息并返回。

3. **调用其他 NCCL 函数的结果处理：**
   ```c
   NCCLCHECK(bootstrapGetUniqueId(&handle));
   ```
   如果获取 unique ID 的操作失败，也会立即退出并返回错误。

---

## **总结**
`NCCLCHECK` 是一个宏，用于简化和统一错误处理的逻辑。它的主要功能是：
1. 调用 NCCL 函数并捕获返回值。
2. 检查返回值是否成功。
3. 如果失败，打印调试信息，并退出当前函数。

# cudaSetDevice
```c
cudaError_t cudaSetDevice(int device);
```
* 其实这并不是一个NCCL的函数，而是一个CUDA runtime的API
* 用于设置当前线程的CUDA设备(GPU)
* 就是说，我现在如果调用了cudaSetDevice(1)，那么接下来的CUDA函数调用都会在GPU 1上执行（我在操作1号设备），直到我再次对另一个设备调用cudaSetDevice

# cudaMalloc
```c
cudaError_t cudaMalloc(void** devPtr, size_t size);
```
* 为设备分配内存，这个设备就是之前用cudaSetDevice设置的设备
* devPtr是一个指向指针的指针，指向的指针存的是分配的内存的地址
* 举例：
```c
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
```

<img src="./NCCL代码中常用的函数和宏定义/3a3ee784d1c88e46f7bd139614358f46.jpg" width="50%">


# group API

### 1. **Group Calls (组调用) 的概念**

`ncclGroupStart()` 和 `ncclGroupEnd()` 是 NCCL 提供的两个函数，用于将多个 NCCL 操作合并成一个操作进行执行。这些操作会在同一个 **NCCL group** 内顺序执行，从而减少了多次启动 NCCL 操作时的开销。通过使用组调用，NCCL 可以更高效地管理并发操作，尤其是在涉及多个 GPU 或多线程的场景下。

- **`ncclGroupStart()`**：启动一个 NCCL 操作组。所有在这个调用后到 `ncclGroupEnd()` 之前的 NCCL 操作都会被视作同一个组的一部分。
  
- **`ncclGroupEnd()`**：结束 NCCL 操作组，并提交所有在 `ncclGroupStart()` 和 `ncclGroupEnd()` 之间的操作。调用这个函数后，NCCL 会将所有操作打包在一起，并尽可能高效地执行。

### 2. **Group Calls 的使用场景**

- **管理多个 GPU**：在单线程管理多个 GPU 时，使用组调用可以避免死锁，并提高多 GPU 操作的并行性。例如，多个 `ncclAllReduce` 操作可以并行执行，而不是一个接一个地执行，这样可以减少执行延迟。

- **聚合通信操作**：通过将多个 NCCL 集体通信操作合并到一个 NCCL 调用组中，可以显著提高性能，减少启动多个 NCCL 操作时的延迟。

- **合并多个点对点通信操作**：点对点通信（例如 `ncclSend` 和 `ncclRecv`）也可以通过组调用合并，减少启动操作时的延迟。

好的，让我们更深入地探讨每个示例，并解释如果不使用组调用（`ncclGroupStart()` 和 `ncclGroupEnd()`），会发生什么，执行的具体过程是怎样的。

### 1. **管理多个 GPU 的通信操作**

#### 示例：多个设备上的 `ncclAllReduce`

假设我们有多个 GPU（例如 4 个），并希望在每个 GPU 上执行相同的 NCCL 操作（例如 `ncclAllReduce`）。不使用 `ncclGroupStart()` 和 `ncclGroupEnd()` 时，你可能会在每个 GPU 上执行一次 `ncclAllReduce` 操作，每次都可能会等待前一个操作完成，这样会增加执行时间和延迟。

##### 使用 `ncclGroupStart()` 和 `ncclGroupEnd()`

```cpp
ncclGroupStart();  // 开始一个 NCCL 操作组
for (int i = 0; i < nLocalDevs; i++) {
  ncclAllReduce(..., comm[i], stream[i]);  // 在多个 GPU 上执行操作
}
ncclGroupEnd();  // 结束并执行所有操作
```

- 在这里，`ncclGroupStart()` 和 `ncclGroupEnd()` 包围了所有的 `ncclAllReduce` 调用，所有操作会被视为同一个组的一部分。
- **执行顺序**：NCCL 会在后台调度这些操作并行执行。每个 GPU 上的操作并不会阻塞其他操作的执行。
- **性能**：所有 GPU 上的 `ncclAllReduce` 操作可以并行执行，减少了同步和启动开销。

##### 不使用 `ncclGroupStart()` 和 `ncclGroupEnd()`

```cpp
for (int i = 0; i < nLocalDevs; i++) {
  ncclAllReduce(..., comm[i], stream[i]);  // 在多个 GPU 上执行操作
}
```

- **执行顺序**：如果不使用组调用，NCCL 会逐个执行这些 `ncclAllReduce` 操作，**等待每个操作完成后再执行下一个操作**。
  - 比如，假设有 4 个 GPU，`ncclAllReduce` 操作会依次执行，每次执行时都会等待前一个操作完成，然后才会开始下一个操作。这种顺序执行会导致明显的延迟。
  
- **死锁风险**：如果在每个操作中都需要同步，且这些操作依赖于其他线程/进程的结果，可能会导致死锁或不必要的阻塞。例如，`ncclAllReduce` 在每个设备上执行时，可能需要等待所有设备的操作完成。如果每个设备操作的顺序不一致，可能导致不必要的等待或冲突。

### 2. **在创建通信器时使用 Group Calls**

#### 示例：在一个线程中管理多个 GPU

假设你有一个线程需要初始化多个 GPU 上的 NCCL 通信器。初始化操作（如 `ncclCommInitRank`）通常是一个阻塞操作，如果不使用 `ncclGroupStart()` 和 `ncclGroupEnd()`，这些初始化操作会依次执行，每次操作都必须等待前一个操作完成，这可能会浪费时间。

##### 使用 `ncclGroupStart()` 和 `ncclGroupEnd()`

```cpp
ncclGroupStart();  // 开始一个 NCCL 操作组
for (int i = 0; i < nLocalDevs; i++) {
  cudaSetDevice(device[i]);  // 设置当前 GPU
  ncclCommInitRank(comms + i, nranks, commId, rank[i]);  // 初始化通信器
}
ncclGroupEnd();  // 结束并执行所有操作
```

- **执行顺序**：`ncclCommInitRank` 调用会并行地在每个 GPU 上执行。
- **性能**：因为通信器的初始化操作是通过组调用来管理的，NCCL 可以在后台并行处理所有设备的初始化操作，而不是一个接一个地执行。这减少了初始化的时间。
  
##### 不使用 `ncclGroupStart()` 和 `ncclGroupEnd()`

```cpp
for (int i = 0; i < nLocalDevs; i++) {
  cudaSetDevice(device[i]);  // 设置当前 GPU
  ncclCommInitRank(comms + i, nranks, commId, rank[i]);  // 初始化通信器
}
```

- **执行顺序**：`ncclCommInitRank` 会按顺序执行，等待每个设备的初始化完成后再继续执行下一个设备的初始化。
- **性能**：没有组调用，通信器初始化会串行执行，导致设备初始化时间更长。如果有多个 GPU，这会浪费很多时间在设备间的同步和等待上。

### 3. **聚合通信操作**

#### 示例：多个集体操作（`ncclBroadcast` 和 `ncclAllReduce`）聚合

假设你想在多个 GPU 上执行多个不同的 NCCL 集体操作（例如，一个 `ncclBroadcast` 和两个 `ncclAllReduce`）。如果不使用组调用，NCCL 会为每个操作单独启动一次通信。

##### 使用 `ncclGroupStart()` 和 `ncclGroupEnd()`

```cpp
ncclGroupStart();  // 开始 NCCL 操作组
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm, stream);
ncclGroupEnd();  // 结束并执行所有操作
```

- **执行顺序**：所有的集体操作（`ncclBroadcast` 和 `ncclAllReduce`）会被合并到一个组中执行。NCCL 会将这些操作作为一个批次提交，减少了每个操作单独启动时的开销。
- **性能**：通过将多个操作合并成一个组，NCCL 只需要发起一次通信并等待完成，从而减少了启动和同步的延迟。

##### 不使用 `ncclGroupStart()` 和 `ncclGroupEnd()`

```cpp
ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm, stream);
ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm, stream);
ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm, stream);
```

- **执行顺序**：每个 `ncclBroadcast` 和 `ncclAllReduce` 操作都会单独执行，NCCL 会依次启动每个操作并等待前一个操作完成。这样，可能会在每个操作之间产生额外的延迟，尤其是在启动多个 NCCL 操作时。
- **性能**：每个操作都会带来额外的启动开销，导致总体性能下降。

### 4. **非阻塞组操作**

在非阻塞模式下，`ncclGroupStart()` 和 `ncclGroupEnd()` 仍然有用。因为非阻塞通信器可能会返回 `ncclInProgress`，表示操作还在进行中。这时，我们需要确保 NCCL 内核已发出操作，才能进行后续操作（例如，等待 CUDA 流同步）。

##### 使用非阻塞组操作

```cpp
ncclGroupStart();  // 开始 NCCL 操作组
for (int g = 0; g < ngpus; g++) {
  ncclAllReduce(sendbuffs[g] + offsets[i], recvbuffs[g] + offsets[i], counts[i], datatype[i], comms[g], streams[g]);
}
ret = ncclGroupEnd();  // 结束操作组

if (ret == ncclInProgress) {
  for (int g = 0; g < ngpus; g++) {
    do {
      ncclCommGetAsyncError(comms[g], &state);  // 检查通信器的异步错误状态
    } while (state == ncclInProgress);
  }
} else if (ret == ncclSuccess) {
  printf("NCCL kernel issue succeeded\n");
} else {
  reportErrorAndRestart();
}

for (int g = 0; g < ngpus; g++) {
  cudaStreamSynchronize(streams[g]);  // 等待所有流同步
}
```

- **执行顺序**：如果 NCCL 操作是非阻塞的，`ncclGroupEnd()` 返回后，不一定所有操作都已经发出。`ncclInProgress` 表示操作仍在后台发出，用户需要检查操作是否完成（通过 `ncclCommGetAsyncError`），然后等待 CUDA 流同步。

##### 不使用非阻塞组操作

```cpp
for (int g = 0; g < ngpus; g++) {
  ncclAllReduce(sendbuffs[g] + offsets[i], recvbuffs[g] + offsets[i], counts[i], datatype[i], comms[g], streams[g]);
}
cudaStreamSynchronize(streams[g]);  // 等待所有流同步
```

- **执行顺序**：每个 `ncclAllReduce` 操作是同步的，会在每个 GPU 上依次执行，直到所有操作完成

在 NCCL 中，"阻塞组"（Blocking Group）和"非阻塞组"（Non-blocking Group）指的是如何处理多个通信操作的并行执行以及如何同步它们。具体来说，这与多个通信操作在 `ncclGroupStart()` 和 `ncclGroupEnd()` 之间的执行方式以及返回值相关。

### 1. **阻塞组（Blocking Group）**

**阻塞组**意味着当调用 `ncclGroupEnd()` 时，NCCL 会**等待**所有在 `ncclGroupStart()` 和 `ncclGroupEnd()` 之间的 NCCL 操作完全完成（包括启动、执行和同步）。在这种模式下，`ncclGroupEnd()` 会在所有操作完成后返回，意味着直到所有操作完成，你才能继续执行后续的代码。

#### 阻塞组的特点：
- 所有组中的 NCCL 操作会按顺序依次发起并等待完成。
- `ncclGroupEnd()` 会**阻塞**直到所有 NCCL 操作都完成。
- 阻塞组适用于你希望在继续执行其他任务之前等待所有 NCCL 操作完成的场景。

**示例**：

```cpp
ncclGroupStart();  // 开始一个 NCCL 操作组
for (int i = 0; i < nLocalDevs; i++) {
  ncclAllReduce(..., comm[i], stream[i]);  // 在多个 GPU 上执行操作
}
ncclGroupEnd();  // 等待所有操作完成
```

在这个例子中，`ncclGroupEnd()` 会阻塞，直到所有的 `ncclAllReduce` 操作完成。这样可以确保在 `ncclGroupEnd()` 返回之前，所有的 NCCL 操作都已经被提交和执行。

### 2. **非阻塞组（Non-blocking Group）**

**非阻塞组**指的是当你调用 `ncclGroupEnd()` 时，NCCL 并不会阻塞直到所有操作完成。相反，`ncclGroupEnd()` 会尽快返回，并表示操作组已被提交，但后台的 NCCL 操作可能仍在执行中。这种模式允许你执行其他任务，同时在后台继续完成 NCCL 操作。

- **非阻塞组**的核心是当 `ncclGroupEnd()` 返回时，NCCL 操作可能仍在后台执行。这时你可以检查操作是否完成（通过查看返回状态或使用异步错误检查），而不是等待它们同步完成。
- 非阻塞组适用于你希望进行并行操作或不希望阻塞主线程的场景，例如，你希望继续执行其他计算或通信操作，而不是等待每个 NCCL 操作完成。

#### 非阻塞组的特点：
- `ncclGroupEnd()` 会尽快返回，不会等待所有操作完成。
- 当组中的操作还在后台执行时，`ncclGroupEnd()` 会返回 `ncclInProgress`，表示操作仍在进行。
- 你可以通过调用 `ncclCommGetAsyncError()` 来检查操作的状态，以便确认 NCCL 操作是否完成。
- 如果使用非阻塞组，通常需要在后续代码中进行同步（如调用 `cudaStreamSynchronize()`）来确保所有操作完成。

**示例**：

```cpp
ncclGroupStart();  // 开始一个 NCCL 操作组
for (int i = 0; i < nLocalDevs; i++) {
  ncclAllReduce(..., comm[i], stream[i]);  // 在多个 GPU 上执行操作
}
ret = ncclGroupEnd();  // 不等待所有操作完成，尽快返回

// 非阻塞操作完成时的处理
if (ret == ncclInProgress) {
  // 检查操作是否完成
  for (int i = 0; i < nLocalDevs; i++) {
    ncclCommGetAsyncError(comm[i], &state);
    while (state == ncclInProgress) {
      // 等待操作完成
      ncclCommGetAsyncError(comm[i], &state);
    }
  }
}
```

在这个例子中，`ncclGroupEnd()` 会尽快返回，并可能返回 `ncclInProgress`，表示 NCCL 操作仍在后台进行。你可以通过调用 `ncclCommGetAsyncError()` 来轮询每个操作的状态，并在操作完成时继续执行后续代码。

### 3. **阻塞与非阻塞的区别**

| 特性                         | 阻塞组（Blocking Group）                                | 非阻塞组（Non-blocking Group）                    |
|------------------------------|--------------------------------------------------------|---------------------------------------------------|
| `ncclGroupEnd()` 返回时的状态 | 阻塞，直到所有操作完成                                  | 不会阻塞，可能返回 `ncclInProgress`（操作仍在进行） |
| 是否等待操作完成              | 是，`ncclGroupEnd()` 会等待所有操作完成               | 否，`ncclGroupEnd()` 会尽快返回                   |
| 操作是否并行进行              | 是，操作仍然会并行进行，但会阻塞等待完成              | 是，操作并行进行，但不等待所有操作完成           |
| 后续操作的执行时机            | 只有当所有 NCCL 操作完成后才能继续执行后续代码        | 可以在后台完成操作时继续执行其他任务             |
| 错误检查                      | 一般不需要异步错误检查，因为操作会同步执行           | 需要通过 `ncclCommGetAsyncError()` 等检查操作状态 |

### 4. **何时使用阻塞组和非阻塞组**

- **阻塞组**：
  - 当你需要确保所有 NCCL 操作在继续执行其他任务之前都已经完成时，使用阻塞组。例如，当你依赖前面的操作结果进行计算时，必须等待所有操作完成。
  - 在大多数情况下，如果你不需要处理并行任务或不需要优化延迟，阻塞组是默认选择。

- **非阻塞组**：
  - 当你希望在 NCCL 操作还在进行时执行其他任务时，使用非阻塞组。例如，你可以在发起多个 NCCL 操作后，继续执行其他计算任务，而不需要等待这些操作完成。
  - 非阻塞组对于需要最大化计算和通信重叠（overlap）的场景非常有用，特别是在需要执行多个独立的计算任务并行时。

### 总结：

- **阻塞组**的 `ncclGroupEnd()` 会等待所有的 NCCL 操作完成后才返回，适合在操作完成后继续执行后续任务。
- **非阻塞组**的 `ncclGroupEnd()` 会尽快返回，不会等待所有操作完成，适合在后台执行操作的同时继续其他任务，或者减少阻塞等待时间。

选择是否使用阻塞组或非阻塞组，取决于你的应用场景和对延迟的容忍度。如果你希望提高并行性和重叠计算与通信，非阻塞组是一个更好的选择；如果你需要确保所有操作完成后再进行下一步，阻塞组则更合适。