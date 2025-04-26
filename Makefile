INCLUDES    := #-I/usr/local/NVIDIA_SDK/C/common/inc
FLAGS       := -Xcompiler -fopenmp -std=c++11 -Wno-deprecated-gpu-targets -lineinfo -gencode arch=compute_75,code=sm_75
NVCC        := nvcc

EXECUTABLE  = btree
CUFILES     = $(wildcard *.cu)
OBJECTS 	= $(CUFILES:.cu=.o)

# Main target
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) --default-stream per-thread $(FLAGS) $(OBJECTS) -o $(EXECUTABLE)

# To obtain object files
%.o: %.cu
	$(NVCC) -c $(FLAGS) $< -o $@ -I../cub/

#$(EXECUTABLE): $(HEADERS) $(CUFILES) $(COBJ)
#	$(NVCC) $(FLAGS) $(INCLUDES) -o $@ $(MAINSRC)

clean:
	rm -f $(EXECUTABLE) *.o
