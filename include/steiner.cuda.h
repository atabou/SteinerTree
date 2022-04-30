
#ifndef STEINER_CUDA_H

    #define STEINER_CUDA_H

    #include "set.cuda.h"
    #include "graph.cuda.h"
    #include "table.cuda.h"
    
    void fill_steiner_dp_table_gpu(cudatable_t* table, cudagraph_t* g, cudaset_t* t, uint32_t t_size, cudatable_t* distances);

#endif
