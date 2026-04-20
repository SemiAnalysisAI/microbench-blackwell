NVCC := /usr/local/cuda-13.2/bin/nvcc
NVCCFLAGS := -std=c++17 -gencode arch=compute_110,code=sm_110

# Config (override via: make CTAS_PER_SM=2 THREADS_PER_BLOCK=128 LOAD_T=float4)
CTAS_PER_SM       ?= 1        # MLA production default
THREADS_PER_BLOCK ?= 64       # MLA: 2 load warps
LOAD_T            ?= float4   # MLA: uint128_t = 16-byte loads

LDGSTS_DEFINES = -DCTAS_PER_SM=$(CTAS_PER_SM) \
                 -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK) \
                 -DLOAD_T=$(LOAD_T)

all: ldgsts_lat.out ldgsts_lat.ptx

ldgsts_lat.out: ldgsts_lat.cu
	$(NVCC) $(NVCCFLAGS) $(LDGSTS_DEFINES) -o $@ $<

ldgsts_lat.ptx: ldgsts_lat.cu
	$(NVCC) $(NVCCFLAGS) $(LDGSTS_DEFINES) -ptx -o $@ $<

bench: ldgsts_lat.out
	sudo /usr/local/cuda-13.2/bin/ncu --clock-control none \
		--metrics sm__cycles_elapsed.avg,sm__sass_inst_executed_op_ldgsts.sum,dram__bytes_read.sum.per_second \
		./ldgsts_lat.out

clean:
	rm -f *.ptx *.out
