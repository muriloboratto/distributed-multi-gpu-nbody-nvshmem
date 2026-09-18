/******************************************************************************
 *
 * Distributed Multi-GPU N-Body Benchmark
 *
 * Description:
 *   Distributed N-Body benchmark for evaluating different communication
 *   libraries in multi-GPU systems.
 *
 *   The application distributes the N-Body computation across multiple GPUs,
 *   using one MPI process per GPU. Each GPU computes the forces associated
 *   with its local subset of bodies while data required from other GPUs is
 *   exchanged through the selected communication library.
 *
 *   Supported communication libraries. The eight-character argument specifies
 *   the communication library used by the benchmark. For example:
 *
 *     MMMMMMMM = MPI
 *     CCCCCCCC = CUDA-Aware MPI
 *     NNNNNNNN = NCCL
 *     SSSSSSSS = NVSHMEM
 *
 * Compilation:
 *
 *   [murilo.boratto@sdumont]$ module load nvshmem/3.1.7_cuda-11.2_sequana
 *
 *   [murilo.boratto@sdumont]$ make
 *
 * Execution:
 *
 *   Example using one node with four GPUs and one MPI process per GPU:
 *
 *   [murilo.boratto@sdumont]$ mpirun -np 1 ./n_body 0 32768 MMMMMMMM \
 *                                  : -np 1 ./n_body 1 32768 MMMMMMMM \
 *                                  : -np 1 ./n_body 2 32768 MMMMMMMM \
 *                                  : -np 1 ./n_body 3 32768 MMMMMMMM
 *
 * Arguments:
 *
 *   ./n_body <device_id> <number_of_bodies> <libraries>
 *
 *   where:
 *
 *     device_id         = CUDA GPU device assigned to the MPI process
 *     number_of_bodies  = total number of bodies (e.g., 32768)
 *     libraries         = communication library combination
 *                         (MMMMMMMM, CCCCCCCC, NNNNNNNN, SSSSSSSS)
 *
 *   In the example above:
 *
 *     MPI Rank 0 -> GPU 0
 *     MPI Rank 1 -> GPU 1
 *     MPI Rank 2 -> GPU 2
 *     MPI Rank 3 -> GPU 3
 *
 *   Therefore, four MPI processes are launched, each associated with one
 *   NVIDIA GPU.
 *
 *   The NVSHMEM implementation uses one-sided GPU communication, asynchronous
 *   data transfers, CUDA streams, and double buffering to overlap
 *   communication with computation.
 *
 *   Numerical validation is performed independently from the benchmark timing
 *   by comparing sampled GPU-computed forces against a CPU reference
 *   implementation.
 *
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include "HPC_api.h"

#define TILE_DIM 256

extern void calculate_force(double *posXl, double *posYl, double *massl, double *forceX, double *forceY,
                            double *auxPosX, double *auxPosY, double *auxMass, int size, int cond, int w, 
                            cudaStream_t stream);

static int all_nvshmem(const char *libraries)
{
    for (int i = 0; i < 8; ++i)
    {
        if (libraries[i] != 'S') 
            return 0;
    }

    return 1;
}

static int next_remote_owner(int start, ProcessInformation proc)
{
    for (int owner = start; owner < proc.numProcesses; ++owner)
    {
        if (owner != proc.currentRank)
            return owner;
    }

    return -1;
}

static void prefetch_owner(MemoryType remotePosX[2], MemoryType remotePosY[2],
                           MemoryType remoteMass[2], MemoryType *posXl,
                           MemoryType *posYl, MemoryType *massl,
                           int buffer, int owner, cudaStream_t stream)
{
    nvshmemx_double_get_nbi_on_stream((double *)remotePosX[buffer].addr_s, (double *)posXl->addr_s, posXl->size, owner, stream);
    nvshmemx_double_get_nbi_on_stream((double *)remotePosY[buffer].addr_s, (double *)posYl->addr_s, posYl->size, owner, stream);
    nvshmemx_double_get_nbi_on_stream((double *)remoteMass[buffer].addr_s, (double *)massl->addr_s, massl->size, owner, stream);
    nvshmemx_quiet_on_stream(stream);
}

static void n_body_nvshmem(MemoryType posX, MemoryType posXl, MemoryType remotePosX[2],
                           MemoryType posY, MemoryType posYl, MemoryType remotePosY[2],
                           MemoryType mass, MemoryType massl, MemoryType remoteMass[2],
                           MemoryType forceX, MemoryType forceXl,
                           MemoryType forceY, MemoryType forceYl,
                           ProcessInformation proc, Communicator comm)
{
    const int local_size = posXl.size;
    const size_t local_bytes = (size_t)local_size * sizeof(double);
    const size_t offset = (size_t)proc.currentRank * local_size;

    cudaEvent_t ready[2];
    cudaEvent_t consumed[2];
    int buffer_used[2] = {0, 0};

    for (int b = 0; b < 2; ++b)
    {
        cudaEventCreateWithFlags(&ready[b], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&consumed[b], cudaEventDisableTiming);
    }

    nvshmem_barrier_all();
    
    nvshmemx_double_get_nbi_on_stream((double *)posXl.addr_s, (double *)posX.addr_s + offset, local_size, 0, comm.comm_stream);
    nvshmemx_double_get_nbi_on_stream((double *)posYl.addr_s, (double *)posY.addr_s + offset, local_size, 0, comm.comm_stream);
    nvshmemx_double_get_nbi_on_stream((double *)massl.addr_s, (double *)mass.addr_s + offset, local_size, 0, comm.comm_stream);
    nvshmemx_quiet_on_stream(comm.comm_stream);

    cudaStreamSynchronize(comm.comm_stream);
    nvshmem_barrier_all();

    cudaMemsetAsync(forceXl.addr_s, 0, local_bytes, comm.compute_stream);
    cudaMemsetAsync(forceYl.addr_s, 0, local_bytes, comm.compute_stream);

    int prefetched_owner = next_remote_owner(0, proc);
    int prefetched_buffer = 0;

    if (prefetched_owner >= 0)
    {
        prefetch_owner(remotePosX, remotePosY, remoteMass, &posXl, &posYl, &massl, prefetched_buffer, prefetched_owner, comm.comm_stream);
        cudaEventRecord(ready[prefetched_buffer], comm.comm_stream);
        buffer_used[prefetched_buffer] = 1;
    }

    for (int owner = 0; owner < proc.numProcesses; ++owner)
    {
        if (owner == proc.currentRank)
        {
            calculate_force((double *)posXl.addr_s,
                            (double *)posYl.addr_s,
                            (double *)massl.addr_s,
                            (double *)forceXl.addr_s,
                            (double *)forceYl.addr_s,
                            (double *)posXl.addr_s,
                            (double *)posYl.addr_s,
                            (double *)massl.addr_s,
                            local_size, 1, TILE_DIM,
                            comm.compute_stream);
            continue;
        }

        const int current = prefetched_buffer;

        cudaStreamWaitEvent(comm.compute_stream, ready[current], 0);

        calculate_force((double *)posXl.addr_s,
                        (double *)posYl.addr_s,
                        (double *)massl.addr_s,
                        (double *)forceXl.addr_s,
                        (double *)forceYl.addr_s,
                        (double *)remotePosX[current].addr_s,
                        (double *)remotePosY[current].addr_s,
                        (double *)remoteMass[current].addr_s,
                        local_size, 0, TILE_DIM,
                        comm.compute_stream);

        cudaEventRecord(consumed[current], comm.compute_stream);

        int next_owner = next_remote_owner(owner + 1, proc);

        if (next_owner >= 0)
        {
            const int next = 1 - current;

            if (buffer_used[next])
                cudaStreamWaitEvent(comm.comm_stream, consumed[next], 0);

            prefetch_owner(remotePosX, remotePosY, remoteMass, &posXl, &posYl, &massl, next, next_owner, comm.comm_stream);
            cudaEventRecord(ready[next], comm.comm_stream);
            buffer_used[next] = 1;

            prefetched_owner = next_owner;
            prefetched_buffer = next;
        }
    }

    cudaStreamSynchronize(comm.compute_stream);

    nvshmemx_double_put_nbi_on_stream((double *)forceX.addr_s + offset, (double *)forceXl.addr_s, local_size, 0, comm.comm_stream);
    nvshmemx_double_put_nbi_on_stream((double *)forceY.addr_s + offset, (double *)forceYl.addr_s, local_size, 0, comm.comm_stream);
    nvshmemx_quiet_on_stream(comm.comm_stream);
    cudaStreamSynchronize(comm.comm_stream);
    nvshmem_barrier_all();

    if (proc.currentRank == 0)
    {
        updateCPUFromSymmetricMemory(&forceX);
        updateCPUFromSymmetricMemory(&forceY);
    }

    for (int b = 0; b < 2; ++b)
    {
        cudaEventDestroy(ready[b]);
        cudaEventDestroy(consumed[b]);
    }
}

inline void n_body(MemoryType posX, MemoryType posXl, MemoryType remotePosX[2],
                   MemoryType posY, MemoryType posYl, MemoryType remotePosY[2],
                   MemoryType mass, MemoryType massl, MemoryType remoteMass[2],
                   MemoryType forceX, MemoryType forceXl,
                   MemoryType forceY, MemoryType forceYl,
                   ProcessInformation proc, Communicator comm,
                   char *communication_libraries)
{
    if (all_nvshmem(communication_libraries))
    {
        n_body_nvshmem(posX, posXl, remotePosX,
                       posY, posYl, remotePosY,
                       mass, massl, remoteMass,
                       forceX, forceXl, forceY, forceYl,
                       proc, comm);
        return;
    }

    SCATTER((OperationType)communication_libraries[0], &posX, &posXl, &proc, &comm);
    SCATTER((OperationType)communication_libraries[1], &posY, &posYl, &proc, &comm);
    SCATTER((OperationType)communication_libraries[2], &mass, &massl, &proc, &comm);

    cudaMemset(forceXl.addr_d, 0, (size_t)forceXl.size * sizeof(double));
    cudaMemset(forceYl.addr_d, 0, (size_t)forceYl.size * sizeof(double));

    for (int owner = 0; owner < proc.numProcesses; ++owner)
    {
        BROADCAST_PROC((OperationType)communication_libraries[3], &posXl, &remotePosX[0], &proc, &comm, owner);
        BROADCAST_PROC((OperationType)communication_libraries[4], &posYl, &remotePosY[0], &proc, &comm, owner);
        BROADCAST_PROC((OperationType)communication_libraries[5], &massl, &remoteMass[0], &proc, &comm, owner);

        calculate_force((double *)posXl.addr_d,
                        (double *)posYl.addr_d,
                        (double *)massl.addr_d,
                        (double *)forceXl.addr_d,
                        (double *)forceYl.addr_d,
                        (double *)remotePosX[0].addr_d,
                        (double *)remotePosY[0].addr_d,
                        (double *)remoteMass[0].addr_d,
                        posX.size / proc.numProcesses,
                        proc.currentRank == owner,
                        TILE_DIM,
                        comm.compute_stream);

        cudaStreamSynchronize(comm.compute_stream);
    }

    GATHER((OperationType)communication_libraries[6], &forceXl, &forceX, &proc, &comm);
    GATHER((OperationType)communication_libraries[7], &forceYl, &forceY, &proc, &comm);

    if (proc.currentRank == 0)
    {
        updateCPUMemory(&forceX);
        updateCPUMemory(&forceY);
    }
}

/***********************************************************************************/

int main(int argc, char *argv[])
{
    double start = 0.0, stop = 0.0;
    double avg_time, local_time, global_time, total_time = 0.0;

    initMultithreading(argc, argv);

    if (argc != 4)
    {
        fprintf(stderr,"Usage: %s <device_id> <number_of_bodies> <8-char configuration>\n Example: %s 0 32768 SSSSSSSS\n", argv[0], argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int deviceId         = atoi(argv[1]);
    int number_of_bodies = atoi(argv[2]);

    if (strlen(argv[3]) != 8)
    {
        fprintf(stderr, "Configuration must contain exactly 8 characters (M/C/N/S).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    char communication_libraries[9];
    strcpy(communication_libraries, argv[3]);

    for (int i = 0; i < 8; ++i)
    {
        if (communication_libraries[i] != 'M' &&
            communication_libraries[i] != 'C' &&
            communication_libraries[i] != 'N' &&
            communication_libraries[i] != 'S')
        {
            fprintf(stderr, "Invalid communication configuration: %s\n", communication_libraries);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    ProcessInformation proc;
    Communicator comm;
    DeviceInformation dev;

    createCommunicator(deviceId, &dev, &comm, &proc);

    if (number_of_bodies <= 0 || number_of_bodies % proc.numProcesses != 0)
    {
        if (proc.currentRank == 0)
            fprintf(stderr, "number_of_bodies must be positive and divisible by the number of MPI ranks.\n");
        
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    printf("n_body node=%s MPI=%d/%d NVSHMEM=%d/%d device=%d:%s libraries=%s\n",
           proc.name,
           proc.currentRank, proc.numProcesses,
           proc.nvshmemPE, proc.nvshmemPEs,
           dev.deviceId, dev.deviceProp.name,
           communication_libraries);

    MemoryType mass, posX, posY, forceX, forceY;
    MemoryType massl, posXl, posYl, forceXl, forceYl;
    MemoryType remotePosX[2], remotePosY[2], remoteMass[2];

    int m  = number_of_bodies;
    int mi = number_of_bodies / proc.numProcesses;

    commMemory(DOUBLE, m,  &mass);
    commMemory(DOUBLE, m,  &posX);
    commMemory(DOUBLE, m,  &posY);
    commMemory(DOUBLE, m,  &forceX);
    commMemory(DOUBLE, m,  &forceY);

    commMemory(DOUBLE, mi, &massl);
    commMemory(DOUBLE, mi, &posXl);
    commMemory(DOUBLE, mi, &posYl);
    commMemory(DOUBLE, mi, &forceXl);
    commMemory(DOUBLE, mi, &forceYl);

    for (int b = 0; b < 2; ++b)
    {
        commMemory(DOUBLE, mi, &remoteMass[b]);
        commMemory(DOUBLE, mi, &remotePosX[b]);
        commMemory(DOUBLE, mi, &remotePosY[b]);
    }

    const int loop_count = 10;

    for (int iteration = 0; iteration < loop_count; iteration++)
    {
        if (proc.currentRank == 0)
        {
            for (int j = 0; j < m; j++)
            {
                ((double *)mass.addr_h)[j]   = (double)j + 1.0;
                ((double *)posX.addr_h)[j]   = (double)j + 1.0;
                ((double *)posY.addr_h)[j]   = (double)j + 1.0;
                ((double *)forceX.addr_h)[j] = 0.0;
                ((double *)forceY.addr_h)[j] = 0.0;
            }

            updateGPUMemory(&mass);
            updateGPUMemory(&posX);
            updateGPUMemory(&posY);
            updateGPUMemory(&forceX);
            updateGPUMemory(&forceY);

            if (all_nvshmem(communication_libraries))
            {
                updateSymmetricMemory(&mass);
                updateSymmetricMemory(&posX);
                updateSymmetricMemory(&posY);
            }
        }

        /* ================================================================ */
        /* n-body                                                           */
        /* ================================================================ */

       //////////////////////////////////////
            MPI_Barrier(MPI_COMM_WORLD);   
               start = MPI_Wtime();       
       //////////////////////////////////////

        n_body(posX, posXl, remotePosX, posY, posYl, remotePosY, mass, massl, remoteMass, forceX, forceXl, forceY, forceYl, proc, comm, communication_libraries);

       //////////////////////////////////////
            MPI_Barrier(MPI_COMM_WORLD);
                stop = MPI_Wtime();          
       //////////////////////////////////////
        
        local_time = stop - start;

        MPI_Reduce(&local_time, 
                   &global_time, 
                   1, 
                   MPI_DOUBLE,
                   MPI_MAX, 
                   0, 
                   MPI_COMM_WORLD);

        if (proc.currentRank == 0)
            total_time += global_time;
    }

    avg_time = total_time / (double)loop_count;

    if (proc.currentRank == 0)
    {
        printf("\nN-Body | Libraries=%s | Bodies=%d | Average Time (s): %8.6f\n", communication_libraries, number_of_bodies, avg_time);

        const int validation_samples = 16;
        const double rel_tolerance = 1.0e-10;
        const double abs_tolerance = 1.0e-12;

    /* ------------------------------------------------------------------ */
    /* Numerical validation                                               */
    /* ------------------------------------------------------------------ */

             validate_nbody_sampled(&posX, &posY, &mass,
                                    &forceX, &forceY,
                                    number_of_bodies,
                                    validation_samples,
                                    rel_tolerance,
                                    abs_tolerance);
        
    }

    /* ------------------------------------------------------------------ */
    /* Cleanup: free only buffers that were actually allocated.           */
    /* ------------------------------------------------------------------ */
    
    freeMemory(&mass);
    freeMemory(&posX);
    freeMemory(&posY);
    freeMemory(&forceX);
    freeMemory(&forceY);

    freeMemory(&massl);
    freeMemory(&posXl);
    freeMemory(&posYl);
    freeMemory(&forceXl);
    freeMemory(&forceYl);

    for (int b = 0; b < 2; ++b)
    {
        freeMemory(&remoteMass[b]);
        freeMemory(&remotePosX[b]);
        freeMemory(&remotePosY[b]);
    }

    destroyComm(&comm);

    return 0;
}
