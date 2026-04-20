NVCCFLAGS := -std=c++17 -gencode arch=compute_110a,code=sm_110a
NVCC := /usr/local/cuda-13.2/bin/nvcc
# Config (override via: make CLUSTER_SIZE=2 SMEM_WIDTH=128 SMEM_HEIGHT=2)
USE_MULTICAST ?= 1
CLUSTER_SIZE ?= 2
SMEM_WIDTH ?= 128
SMEM_HEIGHT ?= 2
NUM_WARPS ?= 4
TMA2DMCAST_DEFINE = -DCLUSTER_SIZE=$(CLUSTER_SIZE) -DSMEM_WIDTH=$(SMEM_WIDTH) -DSMEM_HEIGHT=$(SMEM_HEIGHT) -DUSE_MULTICAST=$(USE_MULTICAST) -DNUM_WARPS=$(NUM_WARPS)

all: tma2dmcast_tput.out tma2dmcast_tput.ptx

tma2dmcast_tput.out: tma2dmcast_tput.cu
	$(NVCC) $(NVCCFLAGS) $(TMA2DMCAST_DEFINE) -lcuda -o $@ $<

tma2dmcast_tput.ptx: tma2dmcast_tput.cu
	$(NVCC) $(NVCCFLAGS) $(TMA2DMCAST_DEFINE) -lcuda -ptx -o $@ $<

bench: tma2dmcast_tput.out
	sudo /usr/local/cuda-13.2/bin/ncu --clock-control none --metrics dram__bytes_read.sum.per_second,sm__cycles_elapsed.avg,sm__sass_inst_executed_op_tma.sum,sm__sass_inst_executed_op_tma.sum.per_cycle_elapsed ./tma2dmcast_tput.out

clean:
	rm -f *.ptx *.out
