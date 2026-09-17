CC = mpic++
NVCC = nvcc

CUDA_DIR = $(CUDA_HOME)
CUDA_INC = -I$(CUDA_DIR)/include -I$(CUDA_DIR)/samples/common/inc
CUDA_LIB = -L$(CUDA_DIR)/lib64

NVSHMEM_DIR = $(NVSHMEM_HOME)
NVSHMEM_INC = -I$(NVSHMEM_DIR)/include
NVSHMEM_LIB = -L$(NVSHMEM_DIR)/lib

NCCL_DIR = $(NCCL_HOME)
NCCL_INC = -I$(NCCL_DIR)/include
NCCL_LIB = -L$(NCCL_DIR)/lib

CUDA_FLAGS = -O2 -rdc=true -Wno-deprecated-gpu-targets \
             -gencode=arch=compute_70,code=sm_70 \
             -Xcompiler -fopenmp

LINK_FLAGS = -rdc=true \
             -gencode=arch=compute_70,code=sm_70 \
             -Xcompiler -fopenmp

LD_LIBS = -lnvidia-ml -lcudart -lnvshmem -lcuda -lnccl -lm

EXE = n_body

OBJ_MAIN = n_body.o
OBJ_API = HPC_api.o
OBJ_CUDA = n_body_kernel.o

OBJS = $(OBJ_MAIN) $(OBJ_API) $(OBJ_CUDA)

$(EXE): $(OBJS)
	$(NVCC) $(LINK_FLAGS) \
	        -ccbin $(CC) \
	        $(OBJS) \
	        -o $(EXE) \
	        $(CUDA_LIB) \
	        $(NVSHMEM_LIB) \
	        $(NCCL_LIB) \
	        $(LD_LIBS)

$(OBJ_MAIN): n_body.c HPC_api.h
	$(NVCC) $(CUDA_FLAGS) \
	        -ccbin $(CC) \
	        $(CUDA_INC) \
	        $(NVSHMEM_INC) \
	        $(NCCL_INC) \
	        -x cu \
	        -c n_body.c \
	        -o $(OBJ_MAIN)

$(OBJ_API): HPC_api.c HPC_api.h
	$(NVCC) $(CUDA_FLAGS) \
	        -ccbin $(CC) \
	        $(CUDA_INC) \
	        $(NVSHMEM_INC) \
	        $(NCCL_INC) \
	        -x cu \
	        -c HPC_api.c \
	        -o $(OBJ_API)

$(OBJ_CUDA): n_body_kernel.cu
	$(NVCC) $(CUDA_FLAGS) \
	        -ccbin $(CC) \
	        $(CUDA_INC) \
	        $(NVSHMEM_INC) \
	        $(NCCL_INC) \
	        -c n_body_kernel.cu \
	        -o $(OBJ_CUDA)

clean:
	rm -f *.o $(EXE)
