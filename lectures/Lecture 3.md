# Fitting it all Together
Once C code is compiled it is then called a "compiled instruction stream" or "scalar instructions".
```
ld r0, addr[r1]
mul r1, r0, r0
add r2, r1, r1
...
```

A **subscalar** processor is capable of executing several of these instructions per clock cycle
- 2 instruction fetch/ decode
- 2 ALU units
```
ld r0, addr[r1]
mul r1, r0, r0 & add r2, r1, r1 <- subscalar interleaved parallelism
```

A **SIMD** processor would have
- 1 instruction fetch/ decode
- 8 ALU units
```
vectorld v0, vector_add[r1]
vector_mul v1, v0, v0
vector_add v
```

A **heterogenous** processor would have
- 2 instruction fetch/ decode
- 9 ALU units (1 scalar ALU, 8 wide ALU)
```
ld r0, addr[r1]
vectorld v0, vector_add[r1]
mul r1, r0, r0 & vector_mul v1, v0, v0 <- subscalar interleaved parallelism
```

A **multi-threaded** proessor would have
- 1 instruction fetch/ decode
- 1 scalar ALU
- 2 execution contexts
```
(execution context 1)
ld r0, addr[r1]
mul r1, r0, r0 <-
add r2, r1, r1
```
```
(execution context 2)
ld r0, addr[r1]
sub r1, r0, r0
add r5, r1, r1 <-
```

A **multi-threaded heterogenous subscalar core** would have
- 2 instruction fetch/ decode
- 9 ALU units (1 scalar ALU, 8 wide ALU)
- 2 execution contexts

We can actually achieve parallelism 2 different ways. Example 1 subscalar parallelism within a single execution context
```
(execution context 1)
vectorld v0, vector_add[r1]
vector_mul v1, v0, v0 & sub r1, r0, r0 <- subscalar interleaved parallelism
vector_add v
```
```
(execution context 2)
vectorld v0, vector_add[r1]
vector_mul v1, v0, v0
sub r1, r0, r0
vector_add v
```

Or subscalar parallelism across 2 execution contexts
```
(execution context 1)
vectorld v0, vector_add[r1] <- simultaneous parallelism
vector_mul v1, v0, v0
vector_add v
```
```
(execution context 2)
vectorld v0, vector_add[r1]
vector_mul v1, v0, v0
sub r1, r0, r0 <- simultaneous parallelism
vector_add v
```

In other words. with such a computer architecture we can execute any 2 steps from any execution context at the same time provided that one of the instructions is vector based.

## SIMT Parallelism
Same Instruction, multiple threads. If the next instruction is the same across 5 execution contexts, then that instruction can be be converted into a single SIMD instruction. This is a common thing in GPU programming.

# Parallel Programming: Abstraction vs Implementation
An important theme is parallel programming is understanding the abstractions already created to allow software engineers to write highly performant code.

## ISPC (Intel SPMD Program Compiler)
SPMD stands for Single Program Multiple Data. Its an abstraction which creates `local` and `uniform` variable states.

### ISPC GANG Abstraction
Each ISPC function will dispatch separate workers to execute the same code across multiple workers with different datapoints.

There are multiple ways to write a SIMT function. Lets consider the first an interleaved assignment stategy

```
export void ispc_sinx(
 uniform int N,
 uniform int terms,
 uniform float* x,
 uniform float* result)
{
 // assume N % programCount = 0
 for (uniform int i=0; i<N; i+=programCount)
 {
    int idx = i + programIndex; // each ISPC worker will have a unique index!
    ...
    for (uniform int j=1; j<=terms; j++)
    {
        ...
    }
    result[idx] = value; // this will hold a unqiue value based upon the program index and the work done at the beginnnig of the process
 }
}
```

Looking at the memory write patterns in `result` array, assuming program count is 3, elements 0 through 2 will be written on the first iteration, 3 through 5 on the next, etc...
```
[0, 0, 0, 1, 1, 1, 2, 2, 2]
```
Our memory access patterns are as follows
```
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
```

Now considered a blocked assignment stategy
```
export void ispc_sinx_v2(
 uniform int N,
 uniform int terms,
 uniform float* x,
 uniform float* result)
{
 uniform int count = N / programCount;
 int start = programIndex * count;

 for (uniform int i=0; i<count; i++)
 {
    int idx = start + i;
    float value = x[idx];
    for (uniform int j=1; j<=terms; j++)
    {
        ...
    }
    result[idx] = value;
 }
}
```
Looking at the memory write patterns in `result` array, the results will now be written in blocks per iteration
```
[0, 1, 2, 0, 1, 2, 0, 1, 2]
```
Our memory access patterns are as follows
```
[0, 3, 6]
[1, 4, 7]
[2, 5, 8]
```

Note how much more efficiently an interleaved ISPC memory access model can be written, where as a block memory access model is much less efficient

**ISPC "Task" Abstraction**
The task abstraction is a separate paradigm which allows us to execute workers across multiple cores

## Shared Address Space Model
Threads share an address space. Within execution context 1, the value at address X is the same value stored at address X in execution context 2. 

### The Problem with a Shared Address Space
Consider 2 threads doing the exact same thing. We are incrementing the same variable in memory w/ 2 separate threads, thus is the initial value was 0, we would expect the result value to be 2.
```
LOAD R1, ADDR[X] // load the variable x
ADD R1, 1 // increment x
LOAD ADDR[X], R1 // write x out to memory
```

The set of instructions might looks as follows
```
thread 1 reads (x = 0)
thread 2 reads (x = 0)
thread 1 increments 
thread 2 increments
thread 1 writes (x = 1)
thread 2 writes (x = 1)
```
Not the expected result! (This kind of error is called a **Race Condition**)

Thus we introduce the concept of locking. If 2 threads are referencing the same item in memory, locking should be used to ensure the expected result. With locking the execution might look like this
```
thread 1 reads (x = 0)
thread 1 increments 
thread 1 writes (x = 1)
thread 2 reads (x = 1)
thread 2 increments 
thread 2 writes (x = 2)
```

### Shared Address Space Computer Architecture
Any core must be able to access any address in memory thus we get these ring like sturctures which connect to each core, and then a separate bus which connects to RAM.

**NUMA** (Non-Uniform Memory Access) is a paradigm which stems from this architecture. Not bandwidth and latency will not be the same across all cores. Some cores will be closer to a certain cache and thus will take less time.

## Message Passing Model
Simple idea. Rather than sharing an address space, a message passing model gives each thread its own private address space. Then if 1 thread needs to communicate its result to another thread, a message passing interface is used.

## Data Parallel Model
Any instance in which the same operation is applied to lots of data. **Sequences** is the name of the data structure over which we can apply such data parallel operations.
