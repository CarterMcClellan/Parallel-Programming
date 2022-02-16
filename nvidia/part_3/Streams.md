# About Streams
Normally when launching a cuda kernel it ends up in the default instruction stream. Thus if several CUDA kernels are launched, each must finish before moving onto the next stage.

Kernels within any single given stream must complete in order. However Kernels in different non-default streams, can interact concurrently. 

Note: The default instruction stream blocks all others, waiting for its tasks to complete before allowing other kernels to resume

## Example Snippet
```cpp
cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.

someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.

cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
```