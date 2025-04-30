INCLUDES    := -I /usr/local/cuda-12/include
FLAGS       := -Xcompiler  -fopenmp -std=c++14 -Wno-deprecated-gpu-targets -lineinfo -gencode=arch=compute_86,code=sm_86
NVCC        := nvcc

EXECUTABLE  = btree
CUFILES     = $(wildcard *.cu)
OBJECTS 	= $(CUFILES:.cu=.o)

# Main target --default-stream per-thread
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) --default-stream per-thread $(FLAGS) $(OBJECTS) -o $(EXECUTABLE)

# To obtain object files
%.o: %.cu
	$(NVCC) -c $(FLAGS) $< -o $@ 

#$(EXECUTABLE): $(HEADERS) $(CUFILES) $(COBJ)
#	$(NVCC) $(FLAGS) $(INCLUDES) -o $@ $(MAINSRC)

clean:
	rm -f $(EXECUTABLE) *.o
