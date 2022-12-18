# Unified Memory Model
- When `cudaMallocManaged` is initial called, the data my not be initially present on the CPU or GPU
- When someone asks for memory the first time, a **page fault** occurs which triggers the migration of demanded memory, onto the CPU.
- This page fault repeats for the GPU and the data is moved there

If we do not want to wait for a page fault to occur because we know what data a machine will need, **asynchronous prefetching** can be used.

## Testing with a Simple Program
We have 2 copies of the exact same function, except one runs on the device, and one runs on the host
```cpp
__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}
```

If we 
- Call only the GPU function
- Call only the CPU function
- Call GPU then CPU
- Call CPU then GPU

How will the outputs from Nsys be different?

(Call only GPU)
```
No output
```

(Call only CPU)
```
No output
```

(Call GPU then CPU)
```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
    89.7         21796655         850  25643.1       63   166270  [CUDA Unified Memory memcpy DtoH]
    10.3          2515866         565   4452.9       63    34495  [CUDA Unified Memory memcpy HtoD]
```

(Call CPU then GPU)
```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         29660157        4825   6147.2     2175   160126  [CUDA Unified Memory memcpy HtoD]
```

## Revisiting SM
Looking at the output results from SM

```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
    76.5         68859156        2304  29886.8     2175   171198  [CUDA Unified Memory memcpy HtoD]
    23.5         21142402         768  27529.2     1183   164446  [CUDA Unified Memory memcpy DtoH]
```

We see that we are wasting a lot of time copying data initialized on Host to the Device. By simply changing that function to run on the device, we can eliminate this time entirely (here is the CUDA Mem Stats after the change)

```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         21271884         768  27697.8     1599   165630  [CUDA Unified Memory memcpy DtoH]
```

## Back to Pre-Fetching

Before Pre-Fetching
```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances   Average    Minimum   Maximum                      Name                    
 -------  ---------------  ---------  ----------  --------  --------  -------------------------------------------
    97.4         62815289          3  20938429.7  17435739  23013147  initWith(float, float*, int)               
     2.6          1708365          1   1708365.0   1708365   1708365  addVectorsInto(float*, float*, float*, int)
```

After Pre-Fetching
```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances   Average   Minimum  Maximum                     Name                    
 -------  ---------------  ---------  ---------  -------  -------  -------------------------------------------
    52.0          1855692          3   618564.0   615833   623226  initWith(float, float*, int)               
    48.0          1712078          1  1712078.0  1712078  1712078  addVectorsInto(float*, float*, float*, int)
```

A ton of time is being saved! Best guess is there are far fewer page faults.


(Before adding Pre-Fetching for CPU)
```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
   100.0         21275773         768  27702.8     1599   170558  [CUDA Unified Memory memcpy DtoH]
```

(After adding Pre-Fetching for the CPU)
```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average   Minimum  Maximum              Operation            
 -------  ---------------  ----------  --------  -------  -------  ---------------------------------
   100.0         20449802          64  319528.2   319036   325308  [CUDA Unified Memory memcpy DtoH]
```

After adding Pre-Fetching for CPU, we can see that the variance in transfer size, and the number of transfers made have each gone significantly down.