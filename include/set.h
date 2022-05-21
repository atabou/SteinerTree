/** 
 * \addtogroup Set
 * @{ */

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
     * @brief Inserts a new integer in the set. If the element already exists nothing is done.
     * 
     * Complexity: O(n)
     * 
     * @param [in] set A pointer to the set to insert in.
     * @param [in] x The element to insert in the set.
     */
    void set_insert(set_t* set, int32_t x);

    /**
     * @brief Checks wether the specified element exists in the set and inside the specified mask.
     * 
     * Complexity: O(n)
     * 
     * @param [in] element The element to search for.
     * @param [in] set A pointer to the set_t to search in.
     * @param [in] mask A mask over which the search is considered.
     * @return int 
     */
    int element_exists(int32_t element, set_t* set, uint64_t mask);

    /**
     * @brief Returns the position of the specified element in the set. If the element does not exist -1 is returned.
     *
     * @param [in] set A pointer to a set_t to search in.
     * @param [in] element The element to search for.
     */
    int32_t find_position(set_t* set, int32_t element);
    
    /**
     * @brief Prints the set in a formatted way.
     * 
     * @param [in] X A pointer to a set_t.
     */
    void print_set(set_t* X);

    /**
     * @brief Destroys and frees the set.
     * 
     * @param [in] set A pointer to a set_t.
     */
    void destroy_set(set_t* set);

#endif
/**@}*/
