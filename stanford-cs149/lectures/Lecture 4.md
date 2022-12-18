# Review from last lectures..
There are 3 abstractions for parallel programming, 

**A Shared Adress Space**
Where the basic idea is that threads can communicate with each other by reading and writing from an address space.
Eg.
```c
int x = 0;

spawn_thread(worker_1, &x)
spawn_thread(worker_2, &x)

void worker_1(int* x) {
    // wait until worker 2 updates the value
    while (x == 0) {}
    printf(x);
}

void worker_2(int* x) {
    x = 1;
}
```

Typically this process will be more involved, and utilize locks. Most programming languages support keywords like atomic, to enable this kinds of interactions.

Note that this all implies that any core should be able to read from any address in memory (otherwise we could not utilize mutiple cores).

**Message Passing**
Basic Idea: Every thread is given their own private address space, threads communicate by sending each other buffers. In this paradigm the only wau to exchange data
is by sending messages.

This model is designed for clusters.

**Data Parallelism**
Whenever you perform the same operation on large amounts of data.

# ISPC Review
The following code has a race condition
```c
foreach (i=0 ... N) {
    if (x[i] < 0) {
        y[i-1] = 2;
    }
    else {
        y[i] = 4;
    }
}
```
Explaination: If the condition is true for (i+1) and false for (i) there is no guarantee that the ith loop will be execute before the (i+1)st (or vice verse),
thus the output is non-deterministic

This is similarly true for a basic summation
```c
uniform float result = 0.0f; // uniform -> shared address space across all threads
foreach (i=0 ... N) {
    result += x[i];
}
```
The sum value will likely be overwritten by multiple threads loading and writing to the same address at the same time.

The solution here would be to create partial sums, contained within each thread
```c
uniform float result = 0.0f; // uniform -> shared address space across all threads

float partial = 0.0f; // exists only within the thread
foreach (i=0 ... N) {
    partial += x[i];
}

sum = reduce_add(partial);
```

Note this is is just a fancy abstraction for the following Intrinsics Code
```c
float tmp[8];
__mm256 partial = __mm256_broadcast_ss(0.0f); // 32 * 8 -> 256 -> 256 = number of bits total, (8 32 bit floats)

for (int i=0; i < N; i+=8) { // we are gonna be loading 8 elements at a time
    partial = __mm256_add_ps(partial, __mm256_load_ps(&x[i]));
}

_mm256_store_ps(tmp); // store the result in temp vector

float result = 0.f;
for (int i=0; i < 8; i++) {
    result += tmp[i];
}

return sum;
```

# Amdahl's Law

## Scenario 1
Parallel Programs are bottlenecked by the parts that need to be executed sequentially. Consider a basic image processing alg.

1. Multiply all pixels by 2.
2. Average the value of all pixels.

Imagine the image is square.

Naively, both ops are N^2, 

step 1 can be run in parallel, so the runtime will go from N^2 to N^2/p
step 2 cannot be made faster

thus the marginal returns in speed as we increase the number of cores in the process are 

2 n^2/(n^2 + n^2/p)

taking the limit of N to infinity, this converges to 2. the greatest speedup we could get on this algorithm is 2x.


## Scenario 2
Reconsider Scenario 1, but this time, lets compute step 2 in parallel

This will be exactly the same as shown above, the partial sum will be done in parallel, the reduce step will be serial, net, the runtime complexit of this
will be n^2/p + p (partial sums + reduce)

thus is marginal returns in speed reduce to

2n^2 / (n^2/p + n^2/p + p)

In this case, speedup converges to P as N -> Infinity

# The big picture
Parallel programming breaks into 3 steps, decomposition, assignment, orchestration, and mapping

**Decomposition**
> Who is responsible for decomposing the program into indepedent tasks?
The programmer. Automated compiler optimizations have failed at this.

**Assignment**
Assignment of "tasks" to "workers"/"threads" is something handled by most languages/ last runtimes.

For instance the ISPC `foreach` keyword we looked at earlier simply picks up incomplete tasks, a programmer does not need to explicitly say
"thread 1 will read x[0..10]", "thread 2 will read x[10..20]", etc...

**Orchestration**
Mapping a thread to a hardware execution context can happen in a couple of ways
- the OS does it
- the compiler does it (eg ISPC will map to a vector instruction lane)
- the hardware does it (a GPU will somehow hand out cuda thread blocks)

There are always some basic goals which we are trying to meet
1. Place related threads on the same processor, (maximize locality, minimize the cost of syncing data)
2. Place unrelated threads on the same processor (if one is read limited and the other is compute limited)