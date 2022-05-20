
#ifndef CUDALLIST_H

    #define CUDALLIST_H

    #include "llist.h"

    typedef llist_t cudallist_t;

    cudallist_t* copy_cudallist(llist_t* lst, int32_t size);
    void free_cudallist(cudallist_t* lst);

#endif
