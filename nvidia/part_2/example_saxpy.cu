#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tid < N )
        c[tid] = 2 * a[tid] + b[tid];
}

__global__ void initMem(int num, float *a, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if ( tid < N )
        a[tid] = num;
}

int main()
{
    // Optimum Block Size
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);
    
    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = 32 * numberOfSMs;
    printf("numberOfBlocks: %lu\nthreadsPerBlock: %lu\n", numberOfBlocks, threadsPerBlock);
    
    // Initialize A, B, C vectors
    float *a, *b, *c;
    int size = N * sizeof (int); // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    
    // Prefetching for GPU Ops
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);
    
    // Initialize Vector Values
    initMem<<<numberOfBlocks, threadsPerBlock>>>(2, a, N);
    initMem<<<numberOfBlocks, threadsPerBlock>>>(1, b, N);
    initMem<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
    cudaDeviceSynchronize();

    // Compute Saxpy
    saxpy <<< numberOfBlocks, threadsPerBlock >>> ( a, b, c );
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    // Prefetch CPU Data
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    
    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
