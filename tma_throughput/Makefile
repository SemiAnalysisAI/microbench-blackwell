NVCCFLAGS := -std=c++17 -gencode arch=compute_110a,code=sm_110a
NVCC := /usr/local/cuda-13.2/bin/nvcc
# Config (override via: make CTAS_PER_SM=2 NUM_STAGES=3 LOAD_SIZE=32768)
CTAS_PER_SM ?= 1
NUM_STAGES ?= 1
LOAD_SIZE ?= 32768
TMA_DEFINE = -DCTAS_PER_SM=$(CTAS_PER_SM) -DNUM_STAGES=$(NUM_STAGES) -DLOAD_SIZE=$(LOAD_SIZE)

all: tma_tput.out tma_tput.ptx

tma_tput.out: tma_tput.cu
	$(NVCC) $(NVCCFLAGS) $(TMA_DEFINE) -o $@ $<

tma_tput.ptx: tma_tput.cu
	$(NVCC) $(NVCCFLAGS) $(TMA_DEFINE) -ptx -o $@ $<

bench: tma_tput.out
	sudo /usr/local/cuda-13.2/bin/ncu --clock-control none --metrics dram__bytes_read.sum.per_second,sm__cycles_elapsed.avg,sm__sass_inst_executed_op_tma.sum,sm__sass_inst_executed_op_tma.sum.per_cycle_elapsed ./tma_tput.out

clean:
	rm -f *.ptx *.out
