
#ifndef CUDASET_H

    #define CUDASET_H

    #include "set.h"

    typedef set_t cudaset_t;

    cudaset_t* copy_cudaset(set_t* set);
    void free_cudaset(cudaset_t* set);

#endif
