#include "HPC_api.h"
#include <string.h>
#include <math.h>

static void copy_d_to_s(MemoryType *mem)
{
    if (mem->size > 0)
        cudaMemcpy(mem->addr_s, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToDevice);
}

static void copy_s_to_d(MemoryType *mem)
{
    if (mem->size > 0)
        cudaMemcpy(mem->addr_d, mem->addr_s, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToDevice);
}

void initMultithreading(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
}

void createCommunicator(int index, DeviceInformation *info, Communicator *comm, ProcessInformation *proc)
{
    comm->MPI_comunicator = MPI_COMM_WORLD;

    MPI_Comm_rank(comm->MPI_comunicator, &proc->currentRank);
    MPI_Comm_size(comm->MPI_comunicator, &proc->numProcesses);
    MPI_Get_processor_name(proc->name, &proc->long_name);

    proc->nvshmemPE  = nvshmem_my_pe();
    proc->nvshmemPEs = nvshmem_n_pes();

    if (proc->nvshmemPE != proc->currentRank || proc->nvshmemPEs != proc->numProcesses)
    {
        fprintf(stderr, "MPI/NVSHMEM mapping mismatch: MPI=%d/%d NVSHMEM=%d/%d\n",
                proc->currentRank, proc->numProcesses,
                proc->nvshmemPE, proc->nvshmemPEs);
       
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (proc->currentRank == 0)
        ncclGetUniqueId(&proc->ncclId);

    MPI_Bcast(&proc->ncclId, sizeof(proc->ncclId), MPI_BYTE, 0, comm->MPI_comunicator);

    info->deviceId = index;
    cudaSetDevice(info->deviceId);
    cudaStreamCreate(&comm->s);
    cudaStreamCreateWithFlags(&comm->compute_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&comm->comm_stream, cudaStreamNonBlocking);
    cudaGetDeviceProperties(&info->deviceProp, info->deviceId);

    ncclCommInitRank(&comm->ncclComunicator, proc->numProcesses, proc->ncclId, proc->currentRank);
}

void commMemory(type tipo, int size, MemoryType *res)
{
    res->size = size;

    switch (tipo)
    {
        case CHAR:
            res->mpi_type = MPI_CHAR;
            res->nccl_type = ncclChar;
            res->bytes_type = sizeof(char);
            break;
        case INT:
            res->mpi_type = MPI_INT;
            res->nccl_type = ncclInt;
            res->bytes_type = sizeof(int);
            break;
        case FLOAT:
            res->mpi_type = MPI_FLOAT;
            res->nccl_type = ncclFloat;
            res->bytes_type = sizeof(float);
            break;
        case DOUBLE:
            res->mpi_type = MPI_DOUBLE;
            res->nccl_type = ncclDouble;
            res->bytes_type = sizeof(double);
            break;
        default:
            fprintf(stderr, "Memory type not implemented\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (size > 0)
    {
        size_t bytes = (size_t)size * res->bytes_type;
        res->addr_h = malloc(bytes);
        cudaMalloc(&res->addr_d, bytes);
        res->addr_s = nvshmem_malloc(bytes);

        if (!res->addr_h || !res->addr_d || !res->addr_s)
        {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    else
    {
        res->addr_h = NULL;
        res->addr_d = NULL;
        res->addr_s = NULL;
    }
}

void updateGPUMemory(MemoryType *mem)
{
    cudaMemcpy(mem->addr_d, mem->addr_h, (size_t)mem->size * mem->bytes_type, cudaMemcpyHostToDevice);
}

void updateCPUMemory(MemoryType *mem)
{
    cudaMemcpy(mem->addr_h, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToHost);
}

void updateSymmetricMemory(MemoryType *mem)
{
    cudaMemcpy(mem->addr_s, mem->addr_h, (size_t)mem->size * mem->bytes_type, cudaMemcpyHostToDevice);
}

void updateCPUFromSymmetricMemory(MemoryType *mem)
{
    cudaMemcpy(mem->addr_h, mem->addr_s, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToHost);
}

void freeMemory(MemoryType *mem)
{
    if (mem->addr_h) free(mem->addr_h);
    if (mem->addr_d) cudaFree(mem->addr_d);
    if (mem->addr_s) nvshmem_free(mem->addr_s);

    mem->addr_h = mem->addr_d = mem->addr_s = NULL;
}

void destroyComm(Communicator *comm)
{
    cudaStreamDestroy(comm->comm_stream);
    cudaStreamDestroy(comm->compute_stream);
    cudaStreamDestroy(comm->s);
    ncclCommDestroy(comm->ncclComunicator);
    nvshmem_finalize();
    MPI_Finalize();
}

void SCATTER(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm)
{
    switch (type)
    {
        case MPI_ONLY:
            if (proc->currentRank == 0)
                cudaMemcpy(mem->addr_h, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToHost);

            MPI_Scatter(mem->addr_h, res->size, res->mpi_type, res->addr_h, res->size, res->mpi_type, 0, comm->MPI_comunicator);
            cudaMemcpy(res->addr_d, res->addr_h, (size_t)res->size * res->bytes_type, cudaMemcpyHostToDevice);
            break;

        case MPI_AWARE:
            MPI_Scatter(mem->addr_d, res->size, res->mpi_type, res->addr_d, res->size, res->mpi_type, 0, comm->MPI_comunicator);
            break;

        case NCCL:
            ncclBroadcast(mem->addr_d, mem->addr_d, mem->size, mem->nccl_type, 0, comm->ncclComunicator, comm->s);
            cudaMemcpyAsync(res->addr_d, (char *)mem->addr_d + (size_t)proc->currentRank * res->size * res->bytes_type, (size_t)res->size * res->bytes_type,
 cudaMemcpyDeviceToDevice, comm->s);
            cudaStreamSynchronize(comm->s);
            break;

        case NVSHMEM:
            if (proc->currentRank == 0)
                copy_d_to_s(mem);

            nvshmem_barrier_all();

            if (res->bytes_type == (int)sizeof(double))
                nvshmem_double_get_nbi((double *)res->addr_s, (double *)mem->addr_s + (size_t)proc->currentRank * res->size, res->size, 0);
            else
            {
                fprintf(stderr, "NVSHMEM path currently implemented for DOUBLE\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            nvshmem_quiet();
            copy_s_to_d(res);
            nvshmem_barrier_all();
            break;

        default:
            fprintf(stderr, "SCATTER: invalid communication type\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void BROADCAST(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm)
{
    BROADCAST_PROC(type, mem, res, proc, comm, 0);
}

void BROADCAST_PROC(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm, int processor_num)
{
    switch (type)
    {
        case MPI_ONLY:
            if (proc->currentRank == processor_num)
                cudaMemcpy(mem->addr_h, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToHost);

            MPI_Bcast(mem->addr_h, mem->size, mem->mpi_type, processor_num, comm->MPI_comunicator);
            cudaMemcpy(res->addr_d, mem->addr_h, (size_t)mem->size * mem->bytes_type, cudaMemcpyHostToDevice);
            break;

        case MPI_AWARE:
            if (proc->currentRank == processor_num && mem != res)
                cudaMemcpy(res->addr_d, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToDevice);

            MPI_Bcast(res->addr_d, res->size, res->mpi_type, processor_num, comm->MPI_comunicator);
            break;

        case NCCL:
            ncclBroadcast(mem->addr_d, res->addr_d, mem->size, mem->nccl_type, processor_num, comm->ncclComunicator, comm->s);
            cudaStreamSynchronize(comm->s);
            break;

        case NVSHMEM:
            copy_d_to_s(mem);
            nvshmem_barrier_all();

            if (proc->currentRank != processor_num)
            {
                nvshmem_double_get_nbi((double *)res->addr_s, (double *)mem->addr_s, mem->size, processor_num);
                nvshmem_quiet();
                copy_s_to_d(res);
            }
            else if (res != mem)
            {
                cudaMemcpy(res->addr_d, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToDevice);
            }

            nvshmem_barrier_all();
            break;

        default:
            fprintf(stderr, "BROADCAST_PROC: invalid communication type\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void GATHER(OperationType type, MemoryType *mem, MemoryType *res, ProcessInformation *proc, Communicator *comm)
{
    switch (type)
    {
        case MPI_ONLY:
            cudaMemcpy(mem->addr_h, mem->addr_d, (size_t)mem->size * mem->bytes_type, cudaMemcpyDeviceToHost);

            MPI_Gather(mem->addr_h, mem->size, mem->mpi_type, res->addr_h, mem->size, res->mpi_type, 0, comm->MPI_comunicator);

            if (proc->currentRank == 0)
                cudaMemcpy(res->addr_d, res->addr_h, (size_t)res->size * res->bytes_type, cudaMemcpyHostToDevice);
            break;

        case MPI_AWARE:
            MPI_Gather(mem->addr_d, mem->size, mem->mpi_type, res->addr_d, mem->size, mem->mpi_type, 0, comm->MPI_comunicator);
            break;

        case NCCL:
            ncclAllGather(mem->addr_d, res->addr_d, mem->size, mem->nccl_type, comm->ncclComunicator, comm->s);
            cudaStreamSynchronize(comm->s);
            break;

        case NVSHMEM:
            copy_d_to_s(mem);
            nvshmem_barrier_all();

            if (proc->currentRank == 0)
            {
                for (int pe = 0; pe < proc->numProcesses; ++pe)
                {
                    nvshmem_double_get_nbi((double *)res->addr_s + (size_t)pe * mem->size, (double *)mem->addr_s, mem->size, pe);
                }
                nvshmem_quiet();
                copy_s_to_d(res);
            }

            nvshmem_barrier_all();
            break;

        default:
            fprintf(stderr, "GATHER: invalid communication type\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}


/* Numerical validation */

void reference_force(const double *posX,
                            const double *posY,
                            const double *mass,
                            int n,
                            int target,
                            double *fx,
                            double *fy)
{
    const double px = posX[target];
    const double py = posY[target];
    long double sumX = 0.0L;
    long double sumY = 0.0L;

    for (int j = 0; j < n; ++j)
    {
        if (j == target)
            continue;

        const long double dx = (long double)px - (long double)posX[j];
        const long double dy = (long double)py - (long double)posY[j];
        const long double d2 = dx * dx + dy * dy;

        if (d2 == 0.0L)
            continue;

        const long double inv_d = 1.0L / sqrtl(d2);
        const long double inv_d3 = inv_d * inv_d * inv_d;

        sumX += dx * (long double)mass[j] * inv_d3;
        sumY += dy * (long double)mass[j] * inv_d3;
    }

    *fx = (double)sumX;
    *fy = (double)sumY;
}

int validate_nbody_sampled(const MemoryType *posX,
                           const MemoryType *posY,
                           const MemoryType *mass,
                           const MemoryType *forceX,
                           const MemoryType *forceY,
                           int number_of_bodies,
                           int max_samples,
                           double rel_tolerance,
                           double abs_tolerance)
{
    const double *x = (const double *)posX->addr_h;
    const double *y = (const double *)posY->addr_h;
    const double *m = (const double *)mass->addr_h;
    const double *gpuFx = (const double *)forceX->addr_h;
    const double *gpuFy = (const double *)forceY->addr_h;

    if (!x || !y || !m || !gpuFx || !gpuFy || number_of_bodies <= 0)
    {
        fprintf(stderr, "VALIDATION ERROR: invalid host data.\n");
        return 0;
    }

    int samples = max_samples;
    if (samples < 1)
        samples = 1;
    if (samples > number_of_bodies)
        samples = number_of_bodies;

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int worst_index = 0;
    int failures = 0;

    long double ref_norm2 = 0.0L;
    long double err_norm2 = 0.0L;

    for (int s = 0; s < samples; ++s)
    {
        const int i = (samples == 1)
                    ? 0
                    : (int)(((long long)s * (number_of_bodies - 1)) / (samples - 1));

        double refFx = 0.0, refFy = 0.0;
        reference_force(x, y, m, number_of_bodies, i, &refFx, &refFy);

        const double ex = gpuFx[i] - refFx;
        const double ey = gpuFy[i] - refFy;
        const double abs_error = hypot(ex, ey);
        const double ref_mag = hypot(refFx, refFy);
        const double rel_error = abs_error / fmax(ref_mag, abs_tolerance);
        const double allowed = abs_tolerance + rel_tolerance * ref_mag;

        ref_norm2 += (long double)refFx * refFx + (long double)refFy * refFy;
        err_norm2 += (long double)ex * ex + (long double)ey * ey;

        if (abs_error > max_abs_error)
        {
            max_abs_error = abs_error;
            worst_index = i;
        }
        if (rel_error > max_rel_error)
            max_rel_error = rel_error;

        if (!isfinite(gpuFx[i]) || !isfinite(gpuFy[i]) || abs_error > allowed)
            ++failures;
    }

    const double relative_l2 = sqrt((double)err_norm2) /
                               fmax(sqrt((double)ref_norm2), abs_tolerance);

    printf("\nNumerical validation (%d sampled bodies)\n", samples);
    printf("  Relative tolerance : %.3e\n", rel_tolerance);
    printf("  Absolute tolerance : %.3e\n", abs_tolerance);
    printf("  Max absolute error : %.6e\n", max_abs_error);
    printf("  Max relative error : %.6e\n", max_rel_error);
    printf("  Relative L2 error  : %.6e\n", relative_l2);
    printf("  Worst sample index : %d\n", worst_index);
    printf("  Result             : %s\n", failures == 0 ? "PASS" : "FAIL");

    return failures == 0;
}
