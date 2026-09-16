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
                    printed = 0;
                }
            }
        }
    }
    return max_diff;
}

template<unsigned int BLOCK_SIZE>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, int M, int N, int K)
{   
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    float* A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K;
    float* B_ptr_start = B_ptr + blockDim.x * blockIdx.x;

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    float temp = 0.0f;
    for(int s = 0; s < K; s += BLOCK_SIZE)
    {
        a_shared[threadIdx.y][threadIdx.x] = A_ptr_start[threadIdx.y * K + s + threadIdx.x];
        b_shared[threadIdx.y][threadIdx.x] = B_ptr_start[(s + threadIdx.y) * N + threadIdx.x];
        __syncthreads();
        for(int k = 0; k < BLOCK_SIZE; k++)
        {
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads();
    }
    C_ptr[x + y * N] = temp;
}

int main()
{
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;
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

    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    // grid(x, y), x is # per row (x-axis), y is # per col(y-axis) but matrix M(i, j) is i in y-axis, j in x-axis
    dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
    cuda_sgemm<BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    if(diff > 0.01f)
    {
        printf("v2: diff not tolerable!\n");
    }
    else
    {
        printf("v2: correct result!\n");
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