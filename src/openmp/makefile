# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -O2 -fopenmp -std=c++17

# Project structure
SRC = main.cpp Vector.cpp Solver/Solver.cpp Solver/ForwardEulerSolver.cpp Solver/LeapFrogSolver.cpp
HEADERS = Vector.h Solver/Solver.h Solver/ForwardEulerSolver.h Solver/LeapFrogSolver.h
OBJS = $(SRC:.cpp=.o)

# Output executable
OUTPUT = ../../build/n-body-openMP

# Default rule
all: $(OUTPUT)

# Linking
$(OUTPUT): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^
	rm -f $(OBJS)

# Compilation
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(OUTPUT)

# Phony targets
.PHONY: all clean
