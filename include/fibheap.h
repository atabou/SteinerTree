
#ifndef FIBHEAP_H

    #define FIBHEAP_H

	#include "heap.h"

    /**
     * @brief Creates a new empty fibonacci heap with a specified initial capacity.
     * 
     * Complexity: O(capacity)
     * 
     * @param capacity the initial capacity of the heap to create. 
     * @return heap* a newly created heap.
     */
    heap_t* make_fibheap(uint32_t capacity);
    

#endif
