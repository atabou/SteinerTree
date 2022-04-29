
#ifndef CUDASET_CUH

    #define CUDASET_CUH

    extern "C" {
        #include "set.h"
    }

    typedef set_t cudaset_t;

    cudaset_t* copy_cudaset(set_t* set);
    void free_cudaset(cudaset_t* set);

#endif
