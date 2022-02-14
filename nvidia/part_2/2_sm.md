# Stream Multiprocessor (SM)
- GPU's are made up of many SM's. 
- Blocks of threads run on SM's. 
    - The number of threads required by a block will effect execution time.
    - Grid dimensions divisible by number of SMs, can promote full SM utilization.
- SMs create, manage, and execute groupings of 32 threads within a block, also known as a warp.

Within C/ C++, the number of SMs on a device can be checked as follows
```
  int deviceId;
  cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about
                                             // the active GPU device.
   
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;

  /*
   * There should be no need to modify the output string below.
   */

  printf("Device ID: %d\nNumber of SMs: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, warpSize);
```

## Optimization
After changing the Cuda Kernel params to
```
  int deviceId;
  cudaGetDevice(&deviceId);                  // `deviceId` now points to the id of the currently active GPU.

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId); // `props` now has many useful properties about
                                             // the active GPU device.
   
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;
  
  threadsPerBlock = warpSize;
  numberOfBlocks = multiProcessorCount;
```

Peformance improves marginally
```
CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum           Name         
 -------  ---------------  ---------  -----------  ---------  ---------  ---------------------
    59.4        228664700          3   76221566.7      17396  228599544  cudaMallocManaged    
    35.2        135486164          1  135486164.0  135486164  135486164  cudaDeviceSynchronize
     5.4         20814881          3    6938293.7    6299715    8052698  cudaFree             
     0.0            42323          1      42323.0      42323      42323  cudaLaunchKernel    
```

```
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                      Name                    
 -------  ---------------  ---------  -----------  ---------  ---------  -------------------------------------------
   100.0        135474363          1  135474363.0  135474363  135474363  addVectorsInto(float*, float*, float*, int)
```

```
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations  Average  Minimum  Maximum              Operation            
 -------  ---------------  ----------  -------  -------  -------  ---------------------------------
    76.5         68859156        2304  29886.8     2175   171198  [CUDA Unified Memory memcpy HtoD]
    23.5         21142402         768  27529.2     1183   164446  [CUDA Unified Memory memcpy DtoH]
```