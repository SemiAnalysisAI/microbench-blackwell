NVCC := /usr/local/cuda-13/bin/nvcc
NVCCFLAGS := -std=c++17 -gencode arch=compute_110a,code=sm_110a

# Defaults
MMA_FORMAT ?= 0
MMA_M ?= 128
MMA_N ?= 64
MMA_K ?= 16
CTA_GROUP ?= 1
AB_LAYOUT ?= 0

DEFINES := -DMMA_FORMAT=$(MMA_FORMAT) -DMMA_M=$(MMA_M) -DMMA_N=$(MMA_N) -DMMA_K=$(MMA_K) \
           -DCTA_GROUP=$(CTA_GROUP) -DAB_LAYOUT=$(AB_LAYOUT)

all: umma_lat.out

umma_lat.out: umma_lat.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -o $@ $<

ptx: umma_lat.cu
	$(NVCC) $(NVCCFLAGS) $(DEFINES) -ptx -o umma_lat.ptx $<

run: umma_lat.out
	./umma_lat.out

clean:
	rm -f *.out *.ptx

.PHONY: all run ptx clean
