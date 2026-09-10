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
{   // first time load global to shared, add two globals to one shared (each block will handle twice the block size when loading from global to shared)
    __shared__ float shared[THREAD_PER_BLOCK]; // volatile ask the compiler to not reuse register as after each step, shared memory has changed but the register is not keeping up.
    float *input_begin = d_input + NUM_PER_BLOCK * blockIdx.x;
    shared[threadIdx.x] = 0;
    // accuracy is bad!!! when too many numbers need to be added to the shared memory. which means num_per_block can not be too big
    for (int i = 0; i < NUM_PER_BLOCK / THREAD_PER_BLOCK; i++)
    {
        shared[threadIdx.x] += input_begin[threadIdx.x + i * THREAD_PER_BLOCK];
    }
    __syncthreads();
    // for (int i = blockDim.x / 2; i > 32; i /= 2)
    // {
    //     if (threadIdx.x < i)
    //     {   
    //         shared[threadIdx.x] += shared[threadIdx.x + i];
    //     }
    //     __syncthreads();
    // }
    // completely unroll the loop
    if (THREAD_PER_BLOCK >= 1024)
    {
        if (threadIdx.x < 512)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 512)
    {
        if (threadIdx.x < 256)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 256)
    {
        if (threadIdx.x < 128)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 128)
    {
        if (threadIdx.x < 64)
        {
            shared[threadIdx.x] += shared[threadIdx.x + 64];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) //when reducing to the last warp, no need to sync between warps anymore
    {
        // shared[threadIdx.x] += shared[threadIdx.x + 32];
        // shared[threadIdx.x] += shared[threadIdx.x + 16];
        // shared[threadIdx.x] += shared[threadIdx.x + 8];
        // shared[threadIdx.x] += shared[threadIdx.x + 4];
        // shared[threadIdx.x] += shared[threadIdx.x + 2];
        // shared[threadIdx.x] += shared[threadIdx.x + 1];
        warpReduce(shared, threadIdx.x);
    }
    if (threadIdx.x == 0)
    {
        d_output[blockIdx.x] = shared[0];
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

    constexpr int block_num = 4096 * 8;
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
        printf("v7 : cpu answer == gpu answer \n");
    }
    else
    {
        printf("v7 : cpu answer != gpu answer \n");
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