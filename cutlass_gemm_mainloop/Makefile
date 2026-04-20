NVCC := /usr/local/cuda-13.2/bin/nvcc
CUTLASS_PATH := ../../cutlass
NVCCFLAGS := -std=c++17 --expt-relaxed-constexpr -gencode=arch=compute_110a,code=\"sm_110a,compute_110a\"
INCLUDES := -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

# Config (override via make TILE_M=128 STAGES=4 etc.)
TILE_M ?= 128
TILE_N ?= 128
TILE_K ?= 64
STAGES ?= 0
DTYPE_ID ?= 0

DEFINES = -DTILE_M=$(TILE_M) -DTILE_N=$(TILE_N) -DTILE_K=$(TILE_K) \
          -DSTAGES=$(STAGES) -DDTYPE_ID=$(DTYPE_ID)

all: cutlass_gemm_bench.out

cutlass_gemm_bench.out: cutlass_gemm_bench.cu
		$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(DEFINES) -lcuda -o $@ $<

bench: cutlass_gemm_bench.out
	./$<

clean:
	rm -f *.out
