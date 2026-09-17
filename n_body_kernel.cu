#include <stdio.h>
#include <cuda_runtime.h>

__global__ void partial_n_body_kernel(double *posX,
                                      double *posY,
                                      double *mass,
                                      double *forceX,
                                      double *forceY,
                                      double *remoteMass,
                                      double *remotePosX,
                                      double *remotePosY,
                                      int size,
                                      int same)
{
    extern __shared__ double shared[];

    double *sPosX = shared;
    double *sPosY = &shared[blockDim.x];
    double *sMass = &shared[2 * blockDim.x];

    int tid = threadIdx.x;
    int part_index = blockIdx.x * blockDim.x + tid;

    double fx = 0.0;
    double fy = 0.0;

    double px = 0.0;
    double py = 0.0;

    if (part_index < size)
    {
        px = posX[part_index];
        py = posY[part_index];
    }

    for (int tile = 0; tile < size; tile += blockDim.x)
    {
        int index = tile + tid;

        if (index < size)
        {
            sPosX[tid] = remotePosX[index];
            sPosY[tid] = remotePosY[index];
            sMass[tid] = remoteMass[index];
        }

        __syncthreads();

        if (part_index < size)
        {
            int tile_size = min((int)blockDim.x, size - tile);

            for (int j = 0; j < tile_size; j++)
            {
                int remote_index = tile + j;

                if (!same || part_index != remote_index)
                {
                    double dx = px - sPosX[j];
                    double dy = py - sPosY[j];

                    double d2 = dx * dx + dy * dy;
                    double inv_d = 1.0 / sqrt(d2);
                    double inv_d3 = inv_d * inv_d * inv_d;

                    fx += dx * sMass[j] * inv_d3;
                    fy += dy * sMass[j] * inv_d3;
                }
            }
        }

        __syncthreads();
    }

    if (part_index < size)
    {
        forceX[part_index] += fx;
        forceY[part_index] += fy;
    }
}


void calculate_force(double *posXl,
                     double *posYl,
                     double *massl,
                     double *forceX,
                     double *forceY,
                     double *auxPosX,
                     double *auxPosY,
                     double *auxMass,
                     int size,
                     int cond,
                     int w,
                     cudaStream_t stream)
{
    dim3 grid1((size + w - 1) / w, 1);
    dim3 block1(w, 1);

    size_t shared_size = 3 * (size_t)w * sizeof(double);

    partial_n_body_kernel<<<grid1, block1, shared_size, stream>>>(posXl,
                                                                  posYl,
                                                                  massl,
                                                                  forceX,
                                                                  forceY,
                                                                  auxMass,
                                                                  auxPosX,
                                                                  auxPosY,
                                                                  size,
                                                                  cond);

    cudaPeekAtLastError();
}
