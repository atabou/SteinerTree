
#include <stdlib.h>

#include "heap.h"
#include "fibheap.h"

heap_t* make_heap(heap_type type, uint32_t capacity) {

    if(type == FIBONACCI) {

        return make_fibheap(capacity);

    } else {

        return NULL;

    }

}
