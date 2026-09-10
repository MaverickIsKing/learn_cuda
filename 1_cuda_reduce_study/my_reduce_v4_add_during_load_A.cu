#include <bits/stdc++.h>
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

// reduce addition kernel static shared memory
__global__ void reduce(float *d_input, float *d_output)
{   // first time load global to shared, add two globals to one shared (each block will handle twice the block size when loading from global to shared)
    __shared__ float shared[THREAD_PER_BLOCK];
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    shared[threadIdx.x] = input_begin[threadIdx.x] + input_begin[threadIdx.x + blockDim.x];
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {   
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads();
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

    int block_num = N / (THREAD_PER_BLOCK * 2);
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
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++)
        {
            current += input[i * 2 * THREAD_PER_BLOCK + j];
        }
        result[i] = current;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    // using gpu
    reduce<<<Grid, Block>>>(d_input, d_output);


    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num))
    {
        printf("v4A : cpu answer == gpu answer \n");
    }
    else
    {
        printf("v4A : cpu answer != gpu answer \n");
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