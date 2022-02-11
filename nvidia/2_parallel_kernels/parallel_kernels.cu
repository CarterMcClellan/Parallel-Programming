#include <stdio.h>

/*
 * Refactor firstParallel so that it can run on the GPU.
 */

__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{

  /*
   * <Number of thread blocks, Number of threads per block>
   */
   firstParallel<<<5, 5>>>();
   cudaDeviceSynchronize();

}