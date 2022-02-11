# nvcc is the command line command for using the nvcc compiler.
# some-CUDA.cu is passed as the file to compile.
# The o flag is used to specify the output file for the compiled program.
# The arch flag indicates for which architecture the files must be compiled. For the present case sm_70 will serve to compile specifically for the GPU this lab is running on, but for those interested in a deeper dive, please refer to the docs about the arch flag, virtual architecture features and GPU features.
# As a matter of convenience, providing the run flag will execute the successfully compiled binary.

nvcc -arch=sm_70 -o hello-gpu 01-hello/01-hello-gpu.cu -run