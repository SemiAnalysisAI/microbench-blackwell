NVCCFLAGS := -std=c++17 -gencode arch=compute_110a,code=sm_110a
NVCC := /usr/local/cuda-13.2/bin/nvcc
# Config (override via: make CTAS_PER_SM=1 NUM_STAGES=4 SMEM_WIDTH=128 SMEM_HEIGHT=2)
CTAS_PER_SM ?= 1
NUM_STAGES ?= 4
SMEM_WIDTH ?= 128
SMEM_HEIGHT ?= 2
TMA2D_DEFINE = -DCTAS_PER_SM=$(CTAS_PER_SM) -DNUM_STAGES=$(NUM_STAGES) -DSMEM_WIDTH=$(SMEM_WIDTH) -DSMEM_HEIGHT=$(SMEM_HEIGHT)

all: tma2d_tput.out tma2d_tput.ptx

tma2d_tput.out: tma2d_tput.cu
	$(NVCC) $(NVCCFLAGS) $(TMA2D_DEFINE) -lcuda -o $@ $<

tma2d_tput.ptx: tma2d_tput.cu
	$(NVCC) $(NVCCFLAGS) $(TMA2D_DEFINE) -lcuda -ptx -o $@ $<

bench: tma2d_tput.out
	sudo /usr/local/cuda-13.2/bin/ncu --clock-control none --metrics dram__bytes_read.sum.per_second,sm__cycles_elapsed.avg,sm__sass_inst_executed_op_tma.avg ./tma2d_tput.out

clean:
	rm -f *.ptx *.out
