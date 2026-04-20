NVCCFLAGS := -std=c++17 -gencode=arch=compute_110,code=\"sm_110,compute_110\" -Xptxas=-v
NVCC := /usr/local/cuda-13.2/bin/nvcc
# Config (override via: make CLUSTER_SIZE=8 THREADS_PER_CTA=1024 LOAD_T=float4 ACCESS_MODE=local STRIDE=1)
CLUSTER_SIZE ?= 8
THREADS_PER_CTA ?= 1024
LOAD_T ?= float4
ACCESS_MODE ?= local
STRIDE ?= 1

# Convert ACCESS_MODE string to numeric value
ifeq ($(ACCESS_MODE),local)
ACCESS_MODE_VAL := 0
else ifeq ($(ACCESS_MODE),bcast)
ACCESS_MODE_VAL := 1
else ifeq ($(ACCESS_MODE),ring)
ACCESS_MODE_VAL := 2
else
$(error ACCESS_MODE must be 'local', 'bcast', or 'ring')
endif

DSMEM_DEFINE = -DCLUSTER_SIZE=$(CLUSTER_SIZE) -DTHREADS_PER_CTA=$(THREADS_PER_CTA) -DLOAD_T=$(LOAD_T) -DACCESS_MODE=$(ACCESS_MODE_VAL) -DSTRIDE=$(STRIDE)

all: dsmem_tput.out

dsmem_tput.out: dsmem_tput.cu common.h
	$(NVCC) $(NVCCFLAGS) $(DSMEM_DEFINE) -o $@ $<

bench: dsmem_tput.out
	sudo /usr/local/cuda-13.2/bin/ncu --clock-control none --metrics sm__sass_data_bytes_mem_shared_op_ld.avg.per_second,sm__cycles_elapsed.avg ./dsmem_tput.out

clean:
	rm -f *.ptx *.out
