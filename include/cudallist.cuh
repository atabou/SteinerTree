
#ifndef CUDALLIST_CUH

#define CUDALLIST_CUH

    extern "C" {
        #include "llist.h"
    }

    typedef llist_t cudallist_t;

    cudallist_t* copy_cudallist(llist_t* lst);
    void free_cudallist(llist_t* lst);

#endif
