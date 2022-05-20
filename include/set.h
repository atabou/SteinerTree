
#ifndef SET_H

    #define SET_H

    #include <stdint.h>

    typedef struct set_t set_t;

    struct set_t {

        int32_t* vals;
        int32_t  size;

    };

    /**
     * @brief Creates a new empty set.
     * 
     * O(1)
     * 
     * @return set_t* the created set.
     */
    set_t* make_set();

    /**
     * @brief Inserts a new integer in the set.
     * 
     * If the element already exists nothing is done.
     * 
     * Complexity: O(n)
     * 
     * @param set the set to insert in.
     * @param x 
     */
    void set_insert(set_t* set, int32_t x);

    /**
     * @brief Checks wether the specified element exists in the set.
     * 
     * Complexity: O(n)
     * 
     * @param element the element to check for.
     * @param set the set
     * @param mask the mask to search in.
     * @return int 
     */
    int element_exists(int32_t element, set_t* set, uint64_t mask);

    int32_t find_position(set_t* set, int32_t element);
    
    /**
     * @brief prints the set in a formatted way.
     * 
     * @param X the set to print.
     */
    void print_set(set_t* X);

    /**
     * @brief Destroys and frees the set.
     * 
     * @param set the set to destroy.
     */
    void destroy_set(set_t* set);

#endif
