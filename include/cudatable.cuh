
#ifndef CUDATABLE_CUH

#define CUDATABLE_CUH

    extern "C" {
        #include "table.h"
    }

    typedef table_t cudatable_t;

    cudatable_t* make_cudatable(uint64_t n, uint64_t m);
    cudatable_t* copy_cudatable(table_t* cpu_table);
    void free_cudatable(cudatable_t* t);



#endif
