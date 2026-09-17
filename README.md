## 1. Overview

![Scheme](img/1.png)

Each body is represented by:

- its X position, `posX`;
- its Y position, `posY`;
- its mass, `mass`.

The resulting force components are stored in:

- `forceX`;
- `forceY`.

For a local body \(i\), the CUDA kernel evaluates its interaction with the bodies \(j\) in the currently available particle partition:


$dx = x_i - x_j$



$dy = y_i - y_j$



$d = \sqrt{dx^2 + dy^2}$


and accumulates:


$F_x \mathrel{+}= \frac{dx\,m_j}{d^3}$



$F_y \mathrel{+}= \frac{dy\,m_j}{d^3}$


When the local and source particle partitions are the same, the interaction of a particle with itself is skipped.

The global particle arrays are divided equally among the MPI processes. Therefore:

```text
local_size = number_of_bodies / number_of_processes
```

The current program requires `number_of_bodies` to be positive and exactly divisible by the number of MPI processes.

---

## 2. Parallel Execution Model

The application follows a **one MPI process per GPU** model.

For a four-GPU execution:

```text
MPI Rank 0  <-->  NVSHMEM PE 0  <-->  GPU 0
MPI Rank 1  <-->  NVSHMEM PE 1  <-->  GPU 1
MPI Rank 2  <-->  NVSHMEM PE 2  <-->  GPU 2
MPI Rank 3  <-->  NVSHMEM PE 3  <-->  GPU 3
```

MPI is initialized first. NVSHMEM is then initialized with `MPI_COMM_WORLD` through:

```cpp
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
```

During communicator creation, the program verifies that:

```text
MPI rank       == NVSHMEM PE
MPI processes  == NVSHMEM PEs
```

If this mapping is not satisfied, the application aborts.

Rank 0 also creates the NCCL unique ID, distributes it with `MPI_Bcast`, and each process initializes its NCCL communicator.

---

## 3. Communication Configuration

The communication mechanism is selected independently for eight communication operations.

The available codes are:

| Code | Communication Mechanism |
| ---- | ----------------------- |
| `M` | MPI |
| `C` | CUDA-Aware MPI |
| `N` | NCCL |
| `S` | NVSHMEM |

The configuration must contain **exactly eight characters**:

```text
communication_libraries[0] -> Scatter posX
communication_libraries[1] -> Scatter posY
communication_libraries[2] -> Scatter mass

communication_libraries[3] -> Exchange posX partitions
communication_libraries[4] -> Exchange posY partitions
communication_libraries[5] -> Exchange mass partitions

communication_libraries[6] -> Gather forceX
communication_libraries[7] -> Gather forceY
```

Examples:

```text
MMMMMMMM -> MPI
CCCCCCCC -> CUDA-Aware MPI
NNNNNNNN -> NCCL
SSSSSSSS -> NVSHMEM
```

Mixed configurations are also accepted by the command-line parser, for example:

```text
CCCSSSCC
SSSNNNSS
```

Each position is interpreted independently by the communication API.

---

## 4. Execution Flow

The N-body execution can be summarized as:

```text
               Global Particle Data
              posX / posY / mass
                       |
                       v
              +----------------+
              |    SCATTER     |
              +----------------+
                       |
                       v
             Local particle data
             posXl / posYl / massl
                       |
                       v
             forceXl = forceYl = 0
                       |
                       v
          +--------------------------+
          | for each partition owner |
          +--------------------------+
                       |
              +--------+--------+
              |                 |
              v                 v
        Local partition    Remote partition
              |                 |
              |           BROADCAST_PROC
              |          or NVSHMEM GET
              |                 |
              +--------+--------+
                       |
                       v
                CUDA N-body kernel
                       |
                       v
             Accumulate local forces
              forceXl / forceYl
                       |
                       v
              +----------------+
              |     GATHER     |
              +----------------+
                       |
                       v
                forceX / forceY
```

The communication can therefore be divided into three principal stages.

### Stage 1 — Initial Particle Distribution

The global arrays:

```text
posX
posY
mass
```

are partitioned among the processes using `SCATTER()`.

Each process obtains:

```text
posXl
posYl
massl
```

with `N/P` elements.

### Stage 2 — Particle-Partition Exchange

Each MPI rank iterates over all partition owners.

For a remote owner, the position and mass arrays are made available through `BROADCAST_PROC()`.

For MPI, CUDA-Aware MPI, and NCCL, this stage uses collective communication semantics.

For NVSHMEM, a process obtains the selected owner's particle partition through a **one-sided GET** operation.

The CUDA kernel then calculates the contribution of that partition to the forces acting on the local bodies.

### Stage 3 — Force Collection

After every particle partition has been processed, the local arrays:

```text
forceXl
forceYl
```

are collected into:

```text
forceX
forceY
```

through `GATHER()`.

---

## 5. Memory Model

`MemoryType` maintains three memory representations:

```cpp
void *addr_h;   // Host memory
void *addr_d;   // CUDA device memory
void *addr_s;   // NVSHMEM symmetric device memory
```

Thus, each allocated data object contains:

```text
Host memory
    |
    | cudaMemcpy
    v
CUDA device memory
    |
    | device-to-device copy when required
    v
NVSHMEM symmetric device memory
```

`commMemory()` allocates:

- host memory with `malloc`;
- conventional GPU memory with `cudaMalloc`;
- symmetric NVSHMEM memory with `nvshmem_malloc`.

The current NVSHMEM implementation intentionally keeps conventional CUDA memory (`addr_d`) separate from symmetric NVSHMEM memory (`addr_s`). Helper functions copy data between these two GPU-memory regions when required.

This preserves the same CUDA computational kernel while introducing NVSHMEM at the communication layer.

---

## 6. Communication Operations

The abstraction is implemented by `HPC_api.c`.

### 6.1 `SCATTER()`

The initial particle distribution differs according to the selected mechanism.

| Code | Implementation |
| ---- | -------------- |
| `M` | Device-to-host copy + `MPI_Scatter` + host-to-device copy |
| `C` | `MPI_Scatter` directly using CUDA device pointers |
| `N` | `ncclBroadcast` of the global array followed by a device-to-device copy of the local partition |
| `S` | PE 0 publishes the global array in symmetric memory and each PE performs `nvshmem_double_get` for its own partition |

For NVSHMEM:

```text
                       PE 0
               Global symmetric array
              +----+----+----+----+
              | D0 | D1 | D2 | D3 |
              +----+----+----+----+
                |    |    |    |
               GET  GET  GET  GET
                |    |    |    |
                v    v    v    v
               PE0  PE1  PE2  PE3
```

Each PE therefore transfers only its own partition.

---

### 6.2 `BROADCAST_PROC()`

This operation makes the particle partition belonging to a selected process available to the other processes.

| Code | Implementation |
| ---- | -------------- |
| `M` | Host/device copies combined with `MPI_Bcast` |
| `C` | `MPI_Bcast` directly using CUDA device memory |
| `N` | `ncclBroadcast` |
| `S` | One-sided `nvshmem_double_get` from the selected PE |

For NVSHMEM, each PE first publishes its local partition at its symmetric address. A consumer that needs the partition of PE `p` executes conceptually:

```text
Consumer PE
    |
    | nvshmem_double_get
    v
Symmetric memory of PE p
```

This is an important difference from the collective mechanisms:

```text
MPI / CUDA-Aware MPI / NCCL
        owner distributes data
                 |
                 v
              receivers
```

whereas the NVSHMEM path uses:

```text
NVSHMEM
       consumer requests remote data
                 |
                 v
          owner's symmetric memory
```

This makes the NVSHMEM implementation particularly useful for experiments involving **data locality** and one-sided GPU communication.

---

### 6.3 `GATHER()`

The force arrays are collected after all particle partitions have been processed.

| Code | Implementation |
| ---- | -------------- |
| `M` | Device-to-host copy + `MPI_Gather` + host-to-device copy on rank 0 |
| `C` | `MPI_Gather` directly using CUDA device memory |
| `N` | `ncclAllGather` |
| `S` | PE 0 performs `nvshmem_double_get` for the local force array of every PE |

The NVSHMEM implementation follows:

```text
PE 0 forceXl ----+
PE 1 forceXl ----+
PE 2 forceXl ----+----> GET by PE 0 ----> forceX
PE 3 forceXl ----+
```

and equivalently for `forceY`.

---

## 7. NVSHMEM Synchronization

The NVSHMEM communication path uses:

```cpp
nvshmem_barrier_all();
nvshmem_quiet();
```

to coordinate publication and completion of symmetric-memory operations.

The general pattern is:

```text
Publish local data
       |
       v
nvshmem_barrier_all()
       |
       v
nvshmem_double_get()
       |
       v
nvshmem_quiet()
       |
       v
Copy symmetric data to CUDA buffer when required
       |
       v
nvshmem_barrier_all()
```

The current implementation uses **host-initiated NVSHMEM operations**. NVSHMEM communication is therefore separated from the CUDA force kernel.

---

## 8. GPU Computation

The CUDA computation is implemented in:

```text
n_body_kernel.cu
```

through:

```cpp
partial_n_body_kernel()
```

and the wrapper:

```cpp
calculate_force()
```

Each CUDA thread processes one local body:

```text
CUDA thread
     |
     v
Local particle i
     |
     +---- interaction with particle 0
     +---- interaction with particle 1
     +---- interaction with particle 2
     +---- ...
     +---- interaction with particle N/P - 1
     |
     v
forceXl[i] / forceYl[i]
```

The kernel accumulates force contributions using:

```cpp
forceX[part_index] += ...
forceY[part_index] += ...
```

For this reason, `n_body()` explicitly resets the local force arrays before processing the partitions:

```cpp
cudaMemset(forceXl.addr_d, 0, ...);
cudaMemset(forceYl.addr_d, 0, ...);
```

When the source partition is the local partition, the `same` argument prevents self-interaction.

The current CUDA block width is:

```cpp
#define TILE_DIM 256
```

and is passed to `calculate_force()`.

> **Note:** despite the name `TILE_DIM`, the current kernel does not implement shared-memory tiling. It is used as the CUDA thread-block width.

---

## 9. Source Files

The project is organized around:

```text
.
├── HPC_api.c
├── HPC_api.h
├── n_body.c
├── n_body_kernel.cu
├── makefile
├── script-execution-1node-4GPUs.sh
└── README.md
```

### `HPC_api.h`

Defines:

- `M`, `C`, `N`, and `S` communication types;
- supported data types;
- host, CUDA, and NVSHMEM symmetric memory descriptors;
- MPI/NCCL communicator information;
- MPI/NVSHMEM process information;
- GPU device information;
- communication API prototypes.

### `HPC_api.c`

Implements:

```text
SCATTER()
BROADCAST()
BROADCAST_PROC()
GATHER()
```

as well as:

- MPI/NVSHMEM initialization;
- MPI/NVSHMEM rank mapping verification;
- NCCL communicator creation;
- host/CUDA/NVSHMEM memory allocation;
- host/device memory transfers;
- NVSHMEM symmetric-memory transfers;
- communicator destruction.

### `n_body.c`

Implements:

- command-line validation;
- process/GPU initialization;
- global and local particle allocation;
- particle initialization;
- initial data distribution;
- local/remote partition traversal;
- force-buffer initialization;
- CUDA-kernel invocation;
- force collection;
- benchmark timing;
- average execution-time calculation.

### `n_body_kernel.cu`

Implements the CUDA N-body force kernel.

### `script-execution-1node-4GPUs.sh`

Runs the supplied four-GPU benchmark configurations for multiple problem sizes.

---

## 10. Command-Line Arguments

The executable receives:

```text
n_body <device_id> <number_of_bodies> <communication_configuration>
```

where:

| Argument | Description |
| -------- | ----------- |
| `device_id` | CUDA device associated with the MPI process |
| `number_of_bodies` | Global number of bodies |
| `communication_configuration` | Exactly eight `M`, `C`, `N`, or `S` characters |

Example:

```bash
./n_body 0 32768 SSSSSSSS
```

corresponds to:

```text
CUDA device       : 0
Number of bodies  : 32768
Communication     : NVSHMEM for all eight communication stages
```

Invalid configuration characters or strings whose length is not eight cause the program to abort.

---

## 11. Requirements

The project requires an HPC environment containing:

- NVIDIA GPUs;
- CUDA Toolkit;
- MPI;
- CUDA-Aware MPI support for `C` experiments;
- NCCL;
- NVSHMEM;
- a C/C++ compiler;
- NVIDIA CUDA compiler (`nvcc`);
- GNU Make or an equivalent build procedure.

Because the communication API includes CUDA, NCCL, and NVSHMEM calls, the build must provide the corresponding header and library paths.

The application links against the NVSHMEM library in addition to MPI, CUDA, and NCCL.

---

## 12. Execution on One Node / Four GPUs

A homogeneous four-GPU MPI execution can be launched using the MPMD syntax:

```bash
mpirun \
  -np 1 ./n_body 0 32768 MMMMMMMM : \
  -np 1 ./n_body 1 32768 MMMMMMMM : \
  -np 1 ./n_body 2 32768 MMMMMMMM : \
  -np 1 ./n_body 3 32768 MMMMMMMM
```

CUDA-Aware MPI:

```bash
mpirun \
  -np 1 ./n_body 0 32768 CCCCCCCC : \
  -np 1 ./n_body 1 32768 CCCCCCCC : \
  -np 1 ./n_body 2 32768 CCCCCCCC : \
  -np 1 ./n_body 3 32768 CCCCCCCC
```

NCCL:

```bash
mpirun \
  -np 1 ./n_body 0 32768 NNNNNNNN : \
  -np 1 ./n_body 1 32768 NNNNNNNN : \
  -np 1 ./n_body 2 32768 NNNNNNNN : \
  -np 1 ./n_body 3 32768 NNNNNNNN
```

NVSHMEM follows the same application-level argument structure:

```bash
mpirun \
  -np 1 ./n_body 0 32768 SSSSSSSS : \
  -np 1 ./n_body 1 32768 SSSSSSSS : \
  -np 1 ./n_body 2 32768 SSSSSSSS : \
  -np 1 ./n_body 3 32768 SSSSSSSS
```

The exact MPI/NVSHMEM launcher requirements may depend on the NVSHMEM and MPI installation used by the target HPC system.

---

## 13. Supplied Benchmark Script

The supplied script currently evaluates:

```text
32768
65536
131072
262144
```

using:

```text
MMMMMMMM
CCCCCCCC
NNNNNNNN
```

on one node with four GPUs.

Its execution pattern is:

```bash
bash script-execution-1node-4GPUs.sh
```

and the results are redirected to separate files for MPI, CUDA-Aware MPI, and NCCL.

Although `n_body.c` and `HPC_api.c` already implement the `S` configuration. It can be added when the NVSHMEM runtime/launcher configuration of the target system has been validated.

---

## 14. Performance Measurement

The application executes:

```text
10 benchmark iterations
```

for each invocation.

Timing is performed around `n_body()` using:

```cpp
MPI_Wtime()
```

The sequence is:

```text
MPI_Barrier
    |
    v
start = MPI_Wtime()
    |
    v
n_body(...)
    |
    v
cudaDeviceSynchronize()
    |
    v
MPI_Barrier
    |
    v
stop = MPI_Wtime()
```

Each process calculates its local elapsed time.

The benchmark then uses:

```cpp
MPI_Reduce(..., MPI_MAX, ...)
```

so rank 0 records the execution time of the slowest participating MPI process for that iteration.

The reported result is the average of the ten reduced execution times:

```text
N-Body | Libraries=SSSSSSSS | Bodies=32768 | Average Time (s): ...
```

---

## 15. Communication and Data-Locality Perspective

The benchmark separates two fundamental components:

```text
N-body computation
        +
communication/data movement
```

The CUDA force kernel remains conceptually the same while the communication mechanism changes.

This enables experiments involving:

- host-mediated MPI communication;
- CUDA-Aware MPI;
- NCCL collective communication;
- NVSHMEM symmetric memory;
- one-sided GPU data access;
- local versus remote particle data;
- communication cost;
- data movement;
- communication/computation balance;
- multi-GPU scaling.

The NVSHMEM path introduces an especially relevant distinction.

With collective communication:

```text
Data owner
    |
    | collective communication
    v
Consumers
```

With the current NVSHMEM implementation:

```text
Consumer
    |
    | one-sided GET
    v
Remote symmetric data
```

Therefore, the benchmark can be used to investigate not only **which communication library is faster**, but also how the **communication model and location of the data** affect effective application performance.

---

## 16. Current NVSHMEM Design

The present implementation should be interpreted as a **NVSHMEM version** designed to preserve the existing CUDA computational kernel.

Its path is conceptually:

```text
CUDA memory
     |
     | device-to-device copy
     v
NVSHMEM symmetric memory
     |
     | nvshmem_double_get
     v
Remote/local symmetric memory
     |
     | device-to-device copy
     v
CUDA memory
     |
     v
N-body CUDA kernel
```

This design has two advantages for experimental comparison:

1. the same N-body CUDA kernel can be preserved;
2. the communication mechanism can be changed independently of the computation.

At the same time, the extra transfers between `addr_d` and `addr_s` are part of the current NVSHMEM implementation and can affect measured performance.

Therefore, results should be interpreted as measurements of the **complete communication strategy implemented by this version**, rather than as a measurement of the raw NVSHMEM transport alone.


---

## 17. Research Motivation

As GPU computational capability increases, effective application performance depends not only on GPU arithmetic throughput, but also on the relationship between:

```text
Where are the data?
        +
How must the data move?
        +
Which communication mechanism is used?
        +
What execution architecture connects the GPUs?
        =
Effective Application Performance
```

The present benchmark provides a controlled environment for studying part of this relationship by keeping the computational workload stable while varying the communication mechanism among:

```text
MPI
CUDA-Aware MPI
NCCL
NVSHMEM
```

This makes the application suitable as a foundation for research on:

**Distributed Multi-GPU Computing · Data Locality · GPU Communication · MPI · CUDA-Aware MPI · NCCL · NVSHMEM · N-Body Simulation · High-Performance Computing**.
