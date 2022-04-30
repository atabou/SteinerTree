
#ifndef CUDATABLE_H

    #define CUDATABLE_H

//    extern "C" {
//        #include "table.h"
//    }

    #include "table.h"

    typedef table_t cudatable_t;

    cudatable_t* make_cudatable(uint64_t n, uint64_t m);
    cudatable_t* copy_cudatable(table_t* cpu_table);
    void free_cudatable(cudatable_t* t);



#endif
