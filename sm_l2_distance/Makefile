NVCC       := nvcc
ARCH       := -gencode=arch=compute_100a,code=\"sm_100a,compute_100a\"
CXXFLAGS   := -O2 -std=c++17
NUM_SMS    ?= 148

TARGET     := l2_pointer_chase
SRC        := l2_pointer_chase.cu

.PHONY: all clean run plot

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(ARCH) -DNUM_SMS=$(NUM_SMS) -o $@ $<

run: $(TARGET)
	mkdir -p results
	./$(TARGET) > results/distance.csv

plot: results/distance.csv
	uv run python3 plot_distance.py results/distance.csv

clean:
	rm -f $(TARGET)
	rm -rf results/
