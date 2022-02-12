
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

prac2:	prac2.cu Makefile
	nvcc prac2.cu -o prac2 $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac2
