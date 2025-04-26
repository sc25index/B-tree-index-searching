GPP        := g++
FLAGS      := -fopenmp -std=c++11

EXECUTABLE  = main
CUFILES     = $(wildcard *.cpp)
OBJECTS 	= $(CUFILES:.cpp=.o)

# Main target
$(EXECUTABLE): $(OBJECTS)
	$(GPP) $(FLAGS) $(OBJECTS) -o $(EXECUTABLE)

# To obtain object files
%.o: %.cpp
	$(GPP) -c $(FLAGS) $< -o $@

#$(EXECUTABLE): $(HEADERS) $(CUFILES) $(COBJ)
#	$(NVCC) $(FLAGS) $(INCLUDES) -o $@ $(MAINSRC)

clean:
	rm -f $(EXECUTABLE) *.o
