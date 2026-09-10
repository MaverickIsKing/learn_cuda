#include <bits/stdc++.h>
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

// reduce addition kernel static shared memory
template<unsigned int NUM_PER_BLOCK>
__global__ void reduce(float *d_input, float *d_output)
{
    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < NUM_PER_BLOCK / THREAD_PER_BLOCK; i++)
    {
        sum += input_begin[threadIdx.x + i * THREAD_PER_BLOCK];
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ float warpLevelSums[32];
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    if (laneId == 0)
    {
        warpLevelSums[warpId] = sum;
    }
    __syncthreads();

    if(warpId == 0)
    {
        sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.0f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    if (threadIdx.x == 0)
    {
        d_output[blockIdx.x] = sum;
    }
}

bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(out[i] - res[i]) > 1e-4f)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    // using pointer of pointer because we need to alter the value of some pointer, we need to pass in the address to that pointer
    // analouge to if we want to change the value of some variable using a function, we need to pass in reference(address) of that variable (now our variable is a pointer)
    cudaMalloc((void**)&d_input, N * sizeof(float));

    constexpr int block_num = 1024 * 32;
    constexpr unsigned int num_per_block = N / block_num;

    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    // using cpu
    for (int i = 0; i < block_num; i++)
    {
        float current = 0;
        for (int j = 0; j < num_per_block; j++)
        {
            current += input[i * num_per_block + j];
        }
        result[i] = current;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    // using gpu
    reduce<num_per_block><<<Grid, Block>>>(d_input, d_output);


    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num))
    {
        printf("v8 : cpu answer == gpu answer \n");
    }
    else
    {
        printf("v8 : cpu answer != gpu answer \n");
        for (int i = 0; i < block_num; i++)
            {
            printf("%lf", output[i]);
            }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}