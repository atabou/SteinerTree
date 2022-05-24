
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
- `Anaconda` or `Miniconda`

## Setup

```
conda create --name cugraphenv
conda activate cugraphenv
conda install -c nvidia -c conda-forge -c rapidsai libcugraph=22.04.00=cuda11_g58be5b53_0
```

## Build & Run: (Only tested on Linux)

### Run tests:

```
cd /path/to/folder
make test
./bin/test
```

### Run main:

```
cd /path/to/folder
make
./bin/main
```

### Other commands: 

- `make clean`
