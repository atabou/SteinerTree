
#ifndef STEINER2_CUDA_H

    #define STEINER2_CUDA_H

    #include "set.cuda.h"
    #include "graph.cuda.h"
    #include "table.cuda.h"

    void steiner_tree_gpu(cudatable_t* table, cudagraph_t* g, uint64_t g_size, cudaset_t* t, uint64_t t_size, cudatable_t* distances);


#endif
