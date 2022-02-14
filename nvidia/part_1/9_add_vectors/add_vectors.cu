#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  // gridDim is the number of blocks on the grid
  int gridStride = gridDim.x * blockDim.x;
  
  for (; i < N; i += gridStride)
  {
    if (i < N){
        result[i] = a[i] + b[i];
    }
  }
  
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  // these are some big ass vectors
  const int N = 2<<20;
  size_t size = N * sizeof(float);
  printf("Dim of vectors is %d\n", N);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);
  
  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;
  addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
  cudaDeviceSynchronize();

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
