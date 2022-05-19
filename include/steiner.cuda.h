
#ifndef STEINER2_CUDA_H

    #define STEINER2_CUDA_H

    #include "set.cuda.h"
    #include "graph.cuda.h"
    #include "table.cuda.h"

    void fill_steiner_dp_table_gpu_2(cudatable_t* table, cudagraph_t* g, uint64_t g_size, cudaset_t* t, uint64_t t_size, cudatable_t* distances);


#endif