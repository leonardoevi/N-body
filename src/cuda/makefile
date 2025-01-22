# Compiler
NVCC = nvcc

# Directories
SRC_DIR = .
FUTILS_DIR = $(SRC_DIR)/futils
INCLUDE_DIR = $(SRC_DIR)/include
KERNELS_DIR = $(SRC_DIR)/kernels
SYSTEM_DIR = $(SRC_DIR)/system

# Files
MAIN = $(SRC_DIR)/main.cpp
FUTILS = $(FUTILS_DIR)/futils.cpp
KERNELS = $(KERNELS_DIR)/kernels.cu
SYSTEM = $(SYSTEM_DIR)/System.cu \
         $(SYSTEM_DIR)/system_friend.cpp \
         $(SYSTEM_DIR)/SystemFE.cu \
         $(SYSTEM_DIR)/SystemLF.cu

# Output
OUTPUT = ../../build/n-body-cuda

# Compiler Flags
NVCCFLAGS = -std=c++17 -I$(INCLUDE_DIR) -I$(FUTILS_DIR) -I$(KERNELS_DIR) -I$(SYSTEM_DIR)

# Targets
all: $(OUTPUT)

# Linker Step
$(OUTPUT): $(MAIN) $(FUTILS) $(KERNELS) $(SYSTEM)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Clean
clean:
	rm -f $(OUTPUT)
