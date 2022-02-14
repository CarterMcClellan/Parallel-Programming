# N Sys

We have a simple program
```cpp
/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

...
addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
```

And we increment `threadsPerBlock` from `1` to `1024`. Using **N Sys** we see the following changes

(Before)
```
CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls    Average      Minimum     Maximum            Name         
 -------  ---------------  ---------  ------------  ----------  ----------  ---------------------
    89.3       2272873106          1  2272873106.0  2272873106  2272873106  cudaDeviceSynchronize
     9.9        251112639          3    83704213.0       18155   251043612  cudaMallocManaged    
     0.8         20889661          3     6963220.3     6266329     8161611  cudaFree             
     0.0            44684          1       44684.0       44684       44684  cudaLaunchKernel 
```

(After)
```
CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum           Name         
 -------  ---------------  ---------  -----------  ---------  ---------  ---------------------
    63.0        274268340          3   91422780.0      19076  274191438  cudaMallocManaged    
    32.0        139151459          1  139151459.0  139151459  139151459  cudaDeviceSynchronize
     5.0         21975015          3    7325005.0    6446845    8591880  cudaFree             
     0.0            51919          1      51919.0      51919      51919  cudaLaunchKernel    
```

(Before)
```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average      Minimum     Maximum                       Name                    
 -------  ---------------  ---------  ------------  ----------  ----------  -------------------------------------------
   100.0       2272862746          1  2272862746.0  2272862746  2272862746  addVectorsInto(float*, float*, float*, int)
```

(After)
```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                      Name                    
 -------  ---------------  ---------  -----------  ---------  ---------  -------------------------------------------
   100.0        139197810          1  139197810.0  139197810  139197810  addVectorsInto(float*, float*, float*, int)
```

(Before)
```
CUDA Memory Operation Statistics (by size in KiB):

   Total     Operations  Average  Minimum  Maximum               Operation            
 ----------  ----------  -------  -------  --------  ---------------------------------
 393216.000        2304  170.667    4.000  1020.000  [CUDA Unified Memory memcpy HtoD]
 131072.000         768  170.667    4.000  1020.000  [CUDA Unified Memory memcpy DtoH]
```

(After)
```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
    76.4         68810237        2304  29865.6     2175   172927  [CUDA Unified Memory memcpy HtoD]
    23.6         21214401         768  27622.9     1503   165118  [CUDA Unified Memory memcpy DtoH]
```

## Observations
- Memory management increases *significantly* when utilizing more threads!.
- Kernel Statistics remain constant, barring that execution time decreases.
- As the number of threads increase **cudaMalloc** overtakes, **cudaDeviceSynchronize**, as the most time intensive task.

