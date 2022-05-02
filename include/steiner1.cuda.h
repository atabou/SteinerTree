
#ifndef STEINER1_CUDA_H

    #define STEINER1_CUDA_H

    #include "set.cuda.h"
    #include "graph.cuda.h"
    #include "table.cuda.h"
    
    void fill_steiner_dp_table_gpu_1(cudatable_t* table, cudagraph_t* g, cudaset_t* t, uint64_t g_size, uint64_t t_size, cudatable_t* distances);

#endif
