INC        := -I$(CUDA_HOME)/include -I. -I../headers
LIB        := -L$(CUDA_HOME)/lib64 -lcudart -lcurand
EXE        := main
NVCC_FLAGS := -lineinfo -arch=native --ptxas-options=-v --use_fast_math --std=c++20 -O3 --expt-relaxed-constexpr 
OBJ_DIR    := bin
INCL_DIR   := include
SRC_DIR    := src
OBJ        := $(OBJ_DIR)/main.obj 

default: $(EXE)


$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.obj : $(SRC_DIR)/%.cu | $(OBJ_DIR)
	nvcc -I $(INCL_DIR) $(INC) $(NVCC_FLAGS) -dc -o $@ $<

$(EXE): $(OBJ)
	nvcc -I $(INCL_DIR) $(INC) $(NVCC_FLAGS) $(OBJ) -o $(EXE) $(LIB)
	
.PHONY: clean
PTX := $(OBJ_DIR)/main.ptx

ptx: $(PTX)

$(OBJ_DIR)/%.ptx : $(SRC_DIR)/%.cu | $(OBJ_DIR)
	nvcc -I $(INCL_DIR) $(INC) $(NVCC_FLAGS) -ptx -o $@ $<

clean:
ifeq ($(OS),Windows_NT)
	@if exist $(OBJ_DIR)\* del /q $(OBJ_DIR)\*
	@if exist $(EXE).exe del /q $(EXE).exe
else
	@rm -f $(OBJ_DIR)/* $(EXE)
endif
