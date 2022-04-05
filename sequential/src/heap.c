
#include <stdlib.h>

#include "heap.h"
#include "fibheap.h"

heap* make_heap(heap_type type, int capacity) {

    if(type == FIBONACCI) {

        return make_fibheap(capacity);

    } else {

        return NULL;

    }

}