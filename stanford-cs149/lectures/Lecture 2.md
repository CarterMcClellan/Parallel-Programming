# Parallel Programming Lectures 1-2
## ILP (Instruction Level Parallelism)
Consider the operation
```
a = x*x + y*y + z*z
```
An inefficient set of instrctions might look like
```
LOAD R0, X
LOAD R1, Y
LOAD R2, Z

MUL R3, R0, R0
MUL R4, R1, R1
MUL R5, R2, R2

ADD R3, R3, R4
ADD R3, R3, R5
```
Note that these instructions
```
MUL R3, R0, R0
MUL R4, R1, R1
MUL R5, R2, R2
```
Can all be performed in parallel! This is called **Instruction Level Parallelism (ILP)**. 

## SIMD
In a loop, the same arithmetic operations tend to be repeated with different input data.
```
for idx, x_i in [x_1, x_2, ..., x_n]:
	y_i = pow(idx, 2) * pow(x_i, idx)
```

To increase efficiency, we can load the same set of instructions `pow(idx, 2)` and `pow(x_i, idx)` to multiple ALUs and process the data in batches (as a vector).

### Conditionals
One large inhibitor of SIMD logic are conditionals, given that the next batch of data cannot execute until the entire present set is finished. Consider the following snippet:
```
for idx, x_i in [x_1, x_2, ..., x_n]:
	if x_i > 1:
		x_i += 1
		x_i *= 5
	else:
		x_1 -= 1
```
With an input vector `[2, 0, 1, 6, 7]` and a batch size of 5 
```
    2 0 1 6 7
1)  * # # * *
2)  * # # * *
3)  # * * # #
```
For clock cycles 1-2 we were only at 60% ALU util, and for clock cycle 3 we were only at 40% util. If ALU util were at 100% then execution would be coherent.

### SIMD vs Multi-Core
Multi-Core does **not** require the instruction stream to be coherent. Each core can execute any kind of instruction.

## Memory
Accessing memory can be the source of “stalls”, clock cycles in which the CPU is waiting for the data to arrive before it can perform computations. Thus we introduce the concept of a cache. There are different tiers of cache’s the tier indicates how many clock cycles are needed to extract data. Eg, L1 -> very few clock cycles, L3 -> more clock cycles than L1, less than DRAM.

Modern CPU’s also introduce the concept of “pre-fetching”. By analyzing a programs memory access patterns a CPU can make predictions to ensure the data is already in the cache when the program needs it.

Another way in which we speed up work is with multiple threads. Imagine a single thread is sitting waiting for data to work. To avoid wasting clock cycles, introduce another thread and allow it to do work. This is a total paradigm shift! This means that by increasing the time a single thread takes to execute we might be able to keep other threads more busy and finish quickly.

What do we do with a thread while it waits to finish? We create an **execution context**.

## Example) Thread Util
Assume we have identical threads which all perform one operation
- ADD (on an ALU) (3 seconds of latency)
- LOAD (12 seconds of latency)

Thus a core with a single thread is only being utilized for (3) out of (15) clock cycles. 20%.

```
    '1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ', '10', '11', '12', '13', '14', '15'
T1   *     *     *     #     #     #     #     #     #     #     #     #     #     #     #

* -> thread working
# -> thread stalled
```

Now lets introduce another thread
```
    '1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ', '10', '11', '12', '13', '14', '15'
T1   *     *     *     #     #     #     #     #     #     #     #     #     #     #     #
T2   #     #     #     *     *     *     #     #     #     #     #     #     #     #     #
```
Core utlilization is now at 40%. The trend is now obvious. Ramping up to 5 we see

```
    '1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ', '9 ', '10', '11', '12', '13', '14', '15'
T1   *     *     *     #     #     #     #     #     #     #     #     #     #     #     #
T2   #     #     #     *     *     *     #     #     #     #     #     #     #     #     #
T3   #     #     #     #     #     #     *     *     *     #     #     #     #     #     #
T4   #     #     #     #     #     #     #     #     #     *     *     *     #     #     #
T5   #     #     #     #     #     #     #     #     #     #     #     #     *     *     *
```
We observe a second important note: adding more threads will yield no more benefit. Core utilization is already at 100%.

If we were to change the math to 
- ADD (on an ALU) (6 seconds of latency)
- LOAD (12 seconds of latency)

We could then see 2 threads occupy 2/3 of the core, and 3 threads occupy 100%. Thus the ratio
```
ADD Latency/ LOAD Latency
```
Dictates the number of threads required to achieve full core util. Note that in this example we are dealing with 1 thread per clock cycle. In the real world we can execute multiple threads. ("threads per core")

## Fictional Chips
An imaginary chip has
- 16 cores
- 8 SIMD ALUs per core
- 4 Threads per core (can hold 4 execution contexts)

From this we can derive
- 16 Simultaneous Instruction Streams (each core can only do a single simultaneous instruction stream)
- 64 Concurrent Instruction Streams (16 Simul Instruction Streams * 4 Threads Per Core)
- 512 independent pieces of work required to achieve max latency.

## Executive Summary
Efficient programs must have
- Have work which can execute in parallel (multi-core)
- Have items which use the same instructions (SIMD)
- Spend enough time on LOAD to enable interleaving of work.

## Latency vs Bandwidth
Imagining we are trying to drive from San Francisco to Stanford. x is the car
```
SF                        Stanford (50km)
---------------------------------------
x -> 100 km/h
---------------------------------------
```
Throughput = 2 cars per hour.

If we were to introduce more lanes
```
SF                        Stanford (50km)
---------------------------------------
x -> 100 km/h
---------------------------------------
x -> 100 km/h
---------------------------------------
```
Throughput = 4 cars per hour.

If we were to increase the number of cars in each lane
```
SF                        Stanford (50km)
---------------------------------------
x -> (1km) x -> (1km) x -> (1km) x
---------------------------------------
x -> (1km) x -> (1km) x -> (1km) x
---------------------------------------
```
If each car were going 100 km/h and 1 km apart, then 1 car would reach Stanford in 1/100th of an hour in each lane. Thus with 2 lanes throughput would be 200 cars per hour.

- **Bandwidth** in this scenario would be the number of lanes, 2.
- **Latency** would be the time it take to get from SF to Stanford, .5 hours.

## All Processes Are Bandwidth Bound
- The greatest limiting factor for HPC is Bandwidth, simply because it is easy to mask latency issues with multiple threads.

## Definitions
There is an important distinction between instruction latency and instruction throughput
- Latency; the amount of time for an instruction to run to completion. eg multiply might take 4 clock cycles.
- Bandwidth: the amount of data being transferred per cycle.
- Throughput; amount of data being transferred over a set period 