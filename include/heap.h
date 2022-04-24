
#ifndef HEAP_H

    #define HEAP_H

    #include <stdint.h>

    typedef enum heap_type {

        FIBONACCI

    } heap_type;

    typedef struct heap_t heap_t;

    struct heap_t {

        void* data;

        /**
         * @brief Inserts a specified id into the heap_t with a specified priority.
         * If the id was previously inserted, nothing is changed.
         * Complexity: O(1)
         * 
         * @param h the heap_t to insert in.
         * @param id the id to insert.
         * @param key the priority of this id.
         */
        void (*insert)(heap_t* h, uint32_t id, uint32_t key);

        /**
         * @brief Checks wether the heap_t is empty.
         * Complexity: O(1)
         * 
         * @param h a heap_t
         * @return 1 if empty, 0 if not
         */
        int (*empty)(heap_t* h);

        /**
         * @brief Finds the minimum id in the heap_t.
         * If the heap_t is empty, -1 is returned.
         * Complexity: O(1)
         * 
         * @param h a heap_t.
         * @return the id with the smallest key in the heap_t.
         */
        uint32_t  (*find_min)(heap_t* h);

        /**
         * @brief Removes the id with smallest key in the heap_t, and consolidates the heap_t.
         * If the heap_t is empty, -1 is returned.
         * Complexity: O(log(n)) amortized.
         * 
         * @param h a heap_t.
         * @return the id with the smallest key in the heap_t.
         */
        uint32_t  (*extract_min)(heap_t* h);

        heap_t*   (*heap_union)(heap_t* h1, heap_t* h2);

        /**
         * @brief Changes the key of the specified id in the heap_t.
         * If the key does not exists noting is changed.
         * Complexity: O(1) amortized.
         * 
         * @param h a heap_t.
         * @param id the id to modify.
         * @param key the new key of the id.
         */
        void    (*decrease_key)(heap_t* h, uint32_t id, uint32_t key);
        void    (*delete_node)(heap_t* h, uint32_t n);

        /**
         * @brief Destroys they heap_t and frees the memory allocated to it.
         * Complexity: O(1) if the array is empty. O(capacity) if not.
         * 
         * @param h the heap_t to destroy.
         */
        void    (*destroy)(heap_t* h);

    };

    /**
     * Creates a heap_t of the specified type with the specified capacity.
     * The complexity of the operation is of the order O(capacity).
     * 
     * @param type type of the heap_t to be created.
     * @param capacity represents the range of ids that can be inserted in the heap_t (0 to capacity - 1).
     * @return heap_t* a constructed heap_t.
     */
    heap_t* make_heap(heap_type type, uint32_t capacity);

#endif
