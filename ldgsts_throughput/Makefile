NVCCFLAGS := -std=c++17 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_103,code=sm_103

# Config (override via: make CTAS_PER_SM=2 NUM_STAGES=3 THREADS_PER_BLOCK=256 LOAD_T=float4)
CTAS_PER_SM ?= 1
NUM_STAGES ?= 1
THREADS_PER_BLOCK ?= 128
LOAD_T ?= float4
LDGSTS_DEFINE = -DCTAS_PER_SM=$(CTAS_PER_SM) -DNUM_STAGES=$(NUM_STAGES) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) -DLOAD_T=$(LOAD_T)

all: ldgsts_tput.out ldgsts_tput.ptx

ldgsts_tput.out: ldgsts_tput.cu
	nvcc $(NVCCFLAGS) $(LDGSTS_DEFINE) -o $@ $<

ldgsts_tput.ptx: ldgsts_tput.cu
	nvcc $(NVCCFLAGS) $(LDGSTS_DEFINE) -ptx -o $@ $<

bench: ldgsts_tput.out
	ncu --clock-control none --metrics dram__bytes_read.sum.per_second,sm__cycles_elapsed.avg,sm__sass_inst_executed_op_ldgsts.sum,sm__sass_inst_executed_op_ldgsts.sum.per_cycle_elapsed ./ldgsts_tput.out

clean:
	rm -f *.ptx *.out
