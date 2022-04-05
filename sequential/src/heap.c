
#include <stdlib.h>

#include "heap.h"
#include "fibheap.h"

heap* make_heap(heap_type type) {

    if(type == FIBONACCI) {

        return make_fibheap();

    } else {

        return NULL;

    }

}