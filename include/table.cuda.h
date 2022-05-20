
#ifndef CUDATABLE_H

    #define CUDATABLE_H

    #include "table.h"

    typedef table_t cudatable_t;

    cudatable_t* make_cudatable(int32_t n, int32_t m);
    cudatable_t* copy_cudatable(table_t* cpu_table);
    void free_cudatable(cudatable_t* t);



#endif
