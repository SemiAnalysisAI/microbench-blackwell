NVCC := /usr/local/cuda-13/bin/nvcc
NVCCFLAGS := -std=c++17 -gencode arch=compute_110a,code=sm_110a

# Defaults
MMA_FORMAT ?= 0
MMA_M ?= 128
MMA_N ?= 64
MMA_K ?= 16
MMA_DEPTH ?= 256
CTA_GROUP ?= 1
AB_LAYOUT ?= 0

DEFINES := -DMMA_FORMAT=$(MMA_FORMAT) -DMMA_M=$(MMA_M) -DMMA_N=$(MMA_N) -DMMA_K=$(MMA_K) \
           -DMMA_DEPTH=$(MMA_DEPTH) -DCTA_GROUP=$(CTA_GROUP) -DAB_LAYOUT=$(AB_LAYOUT)

all: umma_tput.out

umma_tput.out: umma_tput.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -o $@ $<

ptx: umma_tput.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -ptx -o umma_tput.ptx $<

run: umma_tput.out
	./umma_tput.out

prof: umma_tput.out
	sudo ncu --clock-control none --metrics sm__sass_inst_executed_op_utcmma.sum.per_cycle_elapsed,sm__sass_inst_executed_op_utcmma.sum.pct_of_peak_sustained_active,sm__sass_inst_executed_op_utcmma.sum.pct_of_peak_sustained_elapsed ./umma_tput.out

clean:
	rm -f *.out *.ptx

.PHONY: all run ptx prof clean
