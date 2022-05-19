
# Steiner Tree

## Requirements

### Compilers:
- `gcc` >= 9.3 (only works on gcc and maybe clang)
- `nvcc` >= 11.0

### GPU Requirements:
- CUDA 11.0+
- NVIDIA driver 450.80.02+
- Pascal architecture or better

### Build tools:
- `make`
- `cmake` >= 3.20  
- `doxygen` >= 1.8.11
- `graphviz`

## Build

```
git clone https://github.com/rapidsai/cudagraph.git
cd cudagraph/cpp
export CUDACXX="<path to nvcc>"
mkdir build
```

## Run: (Only tested on ubuntu)

```
cd /path/to/folder
make all
./bin/main
```
