#include <cstdio>
#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]

void random_matrix(int m, int n, float* a)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
#if 1       
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else   
            A(i, j) = (j - i) % 3;
#endif
        }
    }
}

void cpu_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {   
            float temp = 0.0f;
            for (int k = 0; k < K; k++)
            {
                temp += A_ptr[m * K + k] * B_ptr[n + k * N];
            }
            C_ptr[m * N + n] = temp;
        }
    }
}

float compare_matrices(int m, int n, float* a, float *b)
{
    float max_diff = 0.0f, diff;
    int printed = 0;
    for(int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            diff = abs(A(i, j) - B(i, j));
            max_diff = (diff > max_diff ? diff : max_diff);
            if (printed == 0)
            {
                if (max_diff > 0.01f)
                {
                    printf("\nerror at (i: %d, j: %d) diff %f, got %f, expected %f\n", i, j, max_diff, A(i, j), B(i, j));
                    printed = 1;
                }
            }
        }
    }
    return max_diff;
}

// x direction grab 4 elements once (float4)
#define FETCH_FLOAT4(pointerVal) (reinterpret_cast<float4*>(&(pointerVal))[0])
template <unsigned int M_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, int M, int N, int K)
{   
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float* A_ptr_start = A_ptr + blockIdx.y * M_NUM_PER_BLOCK * K;
    float* B_ptr_start = B_ptr + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float temp[NUM_PER_THREAD] = {0.0f};

    for (int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[ty * K + s + tx * NUM_PER_THREAD]);
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD]);
        __syncthreads();
        
        for (int i = 0; i < NUM_PER_THREAD; i++)
        {
            for (int k = 0; k < K_NUM_PER_BLOCK; k++)
            {
                temp[i] += a_shared[ty][k] * b_shared[k][tx * NUM_PER_THREAD + i];
            }
        }
        __syncthreads();
    }
    float* C_ptr_start = C_ptr + (blockIdx.y * M_NUM_PER_BLOCK + ty) * N + blockIdx.x * N_NUM_PER_BLOCK + tx * NUM_PER_THREAD; 
    for (int i = 0; i < NUM_PER_THREAD; i++)
    {
        C_ptr_start[i] = temp[i];
    }
}

int main()
{
    const int m = 1024;
    const int n = 1024;
    const int k = 2048;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float* matrix_A_host = (float*)malloc(mem_size_A);
    float* matrix_B_host = (float*)malloc(mem_size_B);

    float* matrix_C_host_gpu_calc = (float*)malloc(mem_size_C);
    float* matrix_C_host_cpu_calc = (float*)malloc(mem_size_C);
    
    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void**)&matrix_A_device, mem_size_A);
    cudaMalloc((void**)&matrix_B_device, mem_size_B);
    cudaMalloc((void**)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);

    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;
    // float 4 therefore divide by 4
    dim3 block(N_NUM_PER_BLOCK / NUM_PER_THREAD, M_NUM_PER_BLOCK);
    // grid(x, y), x is # per row (x-axis), y is # per col(y-axis) but matrix M(i, j) is i in y-axis, j in x-axis
    dim3 grid(n / N_NUM_PER_BLOCK, m / M_NUM_PER_BLOCK);
    cuda_sgemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    if(diff > 0.01f)
    {
        printf("v4: diff not tolerable!\n");
    }
    else
    {
        printf("v4: correct result!\n");
    }
    
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);

    return 0;
}