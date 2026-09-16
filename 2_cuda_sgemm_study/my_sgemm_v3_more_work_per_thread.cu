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

template<unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, int M, int N, int K)
{   
    constexpr int STEP = BLOCK_SIZE * STRIDE; // STEP = dim of data per block of threads
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float* A_ptr_start = A_ptr + blockDim.y * STRIDE * blockIdx.y * K;
    float* B_ptr_start = B_ptr + blockDim.x * STRIDE * blockIdx.x;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];

    float temp[STRIDE][STRIDE] = {0.0f};

    for(int s = 0; s < K; s += STEP)
    {
        for(int i = 0; i < STRIDE; i++)
        {
            for(int j = 0; j < STRIDE; j++)
            {
                a_shared[ty + i * blockDim.y][tx + j * blockDim.x] = A_ptr_start[(ty + i * blockDim.y) * K + s + tx + j * blockDim.x];
                b_shared[ty + i * blockDim.y][tx + j * blockDim.x] = B_ptr_start[(ty + i * blockDim.y + s) * N + tx + j * blockDim.x];
            }
        }
        __syncthreads();

        for(int i = 0; i < STRIDE; i++)
        {
            for(int j = 0; j < STRIDE; j++)
            {
                for(int k = 0; k < STEP; k++)
                {
                    temp[i][j] += a_shared[ty + i * blockDim.y][k] * b_shared[k][tx + j * blockDim.x];
                }
            }
        }
        __syncthreads();
    }

    float* C_ptr_start = C_ptr + blockIdx.y * STEP * N + blockIdx.x * STEP;
    for(int i = 0; i < STRIDE; i++)
    {
        for(int j = 0; j < STRIDE; j++)
        {
            C_ptr_start[(ty + i * blockDim.y) * N + tx + j * blockDim.x] = temp[i][j];
        }
    }
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
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    // grid(x, y), x is # per row (x-axis), y is # per col(y-axis) but matrix M(i, j) is i in y-axis, j in x-axis
    dim3 grid((n + BLOCK - 1) / BLOCK / STRIDE, (m + BLOCK - 1) / BLOCK / STRIDE);
    cuda_sgemm<BLOCK, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    if(diff > 0.01f)
    {
        printf("v3: diff not tolerable!\n");
    }
    else
    {
        printf("v3: correct result!\n");
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