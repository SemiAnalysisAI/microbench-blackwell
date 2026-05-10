ARCH ?= sm_110a

gpc_query: gpc_query.cu
	nvcc -arch=$(ARCH) -O2 -std=c++17 -o $@ $<

clean:
	rm -f gpc_query

.PHONY: clean
