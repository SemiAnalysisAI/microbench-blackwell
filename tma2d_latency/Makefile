NVCCFLAGS := -std=c++17 -gencode arch=compute_110a,code=sm_110a
NVCC := /usr/local/cuda-13.2/bin/nvcc
# Config (override via: make CTAS_PER_SM=1 SMEM_WIDTH=128 SMEM_HEIGHT=2)
CTAS_PER_SM ?= 1
SMEM_WIDTH ?= 128
SMEM_HEIGHT ?= 2
TMA2D_DEFINE = -DCTAS_PER_SM=$(CTAS_PER_SM) -DSMEM_WIDTH=$(SMEM_WIDTH) -DSMEM_HEIGHT=$(SMEM_HEIGHT)

all: tma2d_lat.out tma2d_lat.ptx

tma2d_lat.out: tma2d_lat.cu
	$(NVCC) $(NVCCFLAGS) $(TMA2D_DEFINE) -lcuda -o $@ $<

tma2d_lat.ptx: tma2d_lat.cu
	$(NVCC) $(NVCCFLAGS) $(TMA2D_DEFINE) -lcuda -ptx -o $@ $<

bench: tma2d_lat.out
	sudo $$(which ncu) --clock-control none --metrics sm__cycles_elapsed.avg,sm__sass_inst_executed_op_tma.avg,dram__bytes_read.sum.per_second ./tma2d_lat.out

clean:
	rm -f *.ptx *.out
