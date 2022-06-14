
# Steiner Tree

## Requirements

### Compilers:
- `gcc` >= 9.3 (only works on gcc and maybe clang)
- `nvcc` >= 11.0

### GPU Requirements:
- CUDA 11.0+
- NVIDIA driver 450.80.02+
- Pascal architecture or newer

### Build tools:
- `make`
- `Anaconda` or `Miniconda`
- `cmake` (optional for non-python install)
- `pip` (optional for non-python install)

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

## Python API

### Build:

- `cd <project directory>`
- `python -m pip install venv`
- `python -m venv .venv && source .venv/bin/activate`
- `pip install wheel`
- `make lib`
- `cd scripts`
- `python setup.py bdist_wheel`
- `pip install dist/pysteiner-0.0.1*.whl --force-reinstall`

### Use:

```
from pysteiner import pygraph
from pysteiner import pyquery
from pysteiner import pysteiner

g = pygraph.graph() # (None -> graph) - Create an empty graph

pygraph.insert_vertex(g) # (graph -> int) - The int is the internal ID of the node you inserted, save it!
pygraph.insert_vertex(g)
pygraph.insert_vertex(g)

pygraph.insert_edge(g, 0, 1, 1.0) # (graph -> int -> int -> float -> None) - Inserts a directed edge between two internal IDs
pygraph.insert_edge(g, 1, 0, 1.0) 
pygraph.insert_edge(g, 2, 0, 1.0)
pygraph.insert_edge(g, 0, 2, 1.0)

q = pyquery.query() # (None -> query) - Create an empty query

pyquery.insert(1) # (int -> None) - Insert a vertex to query for
pyquery.insert(2)

result = pysteiner.steiner(g, q) # (graph -> query -> steiner_result) - Computes the minimum steiner tree of the supplied graph with the supplied query.

subgraph = pysteiner.subgraph(result) # (steiner_result -> graph) - Returns the subgraph extracted from the previous computation

cost = pysteiner.cost(result) # (steiner_result -> int) - Returns the cost of the minimum steiner tree

# Manual memory management is currently required
pygraph.destroy(g)
pygraph.destroy(subgraph)
pyquery.destroy(q)
pysteiner.destroy(result)
```
