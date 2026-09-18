#ifndef HPC_API_H
#define HPC_API_H

#include <mpi.h>
#include <nccl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <stdlib.h>

enum OperationType
{
    MPI_ONLY  = 'M',
    MPI_AWARE = 'C',
    NCCL      = 'N',
    NVSHMEM   = 'S'
};

enum type
{
    CHAR,
    INT,
    FLOAT,
    DOUBLE
};

struct MemoryType
{
    void *addr_h;
    void *addr_d;
    void *addr_s;
    MPI_Datatype mpi_type;
    ncclDataType_t nccl_type;
    int bytes_type;
    int size;
};

struct Communicator
{
    ncclComm_t ncclComunicator;
    MPI_Comm MPI_comunicator;
    cudaStream_t s;
    cudaStream_t compute_stream;
    cudaStream_t comm_stream;
};

struct ProcessInformation
{
    ncclUniqueId ncclId;
    int currentRank;
    int numProcesses;
    int nvshmemPE;
    int nvshmemPEs;
    char name[MPI_MAX_PROCESSOR_NAME];
    int long_name;
};

struct DeviceInformation
{
    int deviceId;
    cudaDeviceProp deviceProp;
};

void initMultithreading(int argc, char *argv[]);
void createCommunicator(int index, DeviceInformation *info, Communicator *comm, ProcessInformation *proc);

void commMemory(type tipo, int size, MemoryType *res);
void updateGPUMemory(MemoryType *mem);
void updateCPUMemory(MemoryType *mem);
void updateSymmetricMemory(MemoryType *mem);
void updateCPUFromSymmetricMemory(MemoryType *mem);
void freeMemory(MemoryType *mem);
void destroyComm(Communicator *comm);

void SCATTER(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm);
void BROADCAST(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm);
void BROADCAST_PROC(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm, int processor_num);
void GATHER(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm);

void reference_force(const double *posX,
                     const double *posY,
                     const double *mass,
                     int n,
                     int target,
                     double *fx,
                     double *fy);

int validate_nbody_sampled(const MemoryType *posX,
                           const MemoryType *posY,
                           const MemoryType *mass,
                           const MemoryType *forceX,
                           const MemoryType *forceY,
                           int number_of_bodies,
                           int max_samples,
                           double rel_tolerance,
                           double abs_tolerance);

#endif
