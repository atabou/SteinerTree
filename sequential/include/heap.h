
#ifndef HEAP_H

    #define HEAP_H

    typedef enum heap_type {

        FIBONACCI

    } heap_type;

    typedef struct heap heap;

    struct heap {

        void* data;

        /**
         * @brief Inserts a specified id into the heap with a specified priority.
         * If the id was previously inserted, nothing is changed.
         * Complexity: O(1)
         * 
         * @param h the heap to insert in.
         * @param id the id to insert.
         * @param key the priority of this id.
         */
        void (*insert)(heap* h, int id, int key);

        /**
         * @brief Checks wether the heap is empty.
         * Complexity: O(1)
         * 
         * @param h a heap
         * @return 1 if empty, 0 if not
         */
        int (*empty)(heap* h);

        /**
         * @brief Finds the minimum id in the heap.
         * If the heap is empty, -1 is returned.
         * Complexity: O(1)
         * 
         * @param h a heap.
         * @return the id with the smallest key in the heap.
         */
        int     (*find_min)(heap* h);

        /**
         * @brief Removes the id with smallest key in the heap, and consolidates the heap.
         * If the heap is empty, -1 is returned.
         * Complexity: O(log(n)) amortized.
         * 
         * @param h a heap.
         * @return the id with the smallest key in the heap.
         */
        int     (*extract_min)(heap* h);

        heap*   (*heap_union)(heap* h1, heap* h2);

        /**
         * @brief Changes the key of the specified id in the heap.
         * If the key does not exists noting is changed.
         * Complexity: O(1) amortized.
         * 
         * @param h a heap.
         * @param id the id to modify.
         * @param key the new key of the id.
         */
        void    (*decrease_key)(heap* h, int id, int key);
        void    (*delete_node)(heap* h, int n);

        /**
         * @brief Destroys they heap and frees the memory allocated to it.
         * Complexity: O(1) if the array is empty. O(capacity) if not.
         * 
         * @param h the heap to destroy.
         */
        void    (*destroy)(heap* h);

    };

    /**
     * Creates a heap of the specified type with the specified capacity.
     * The complexity of the operation is of the order O(capacity).
     * 
     * @param type type of the heap to be created.
     * @param capacity represents the range of ids that can be inserted in the heap (0 to capacity - 1).
     * @return heap* a constructed heap.
     */
    heap* make_heap(heap_type type, int capacity);

#endif