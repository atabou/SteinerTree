
#ifndef SET_H

    #define SET_H

    typedef struct set_t set_t;

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
    void set_insert(set_t* set, int x);

    /**
     * @brief Get the ith element in the set.
     * 
     * Complexity: O(1)
     * 
     * @param X the set to extract from.
     * @param i the position to extract from.
     * @return int 
     */
    int get_element(set_t* X, int i);

    /**
     * @brief Gets the size of the set.
     * 
     * @param X the set to get the size from.
     * @return int the size of the set.
     */
    int set_size(set_t* X);

    /**
     * @brief Checks wether the specified element exists in the set.
     * 
     * Complexity: O(n)
     * 
     * @param element the element to check for.
     * @param X the set 
     * @return int 
     */
    int element_exists(int element, set_t* X);
    
    /**
     * @brief Removes a speified element from the set.
     * 
     * If the element does not exist nothing is done.
     * 
     * Complexity: O(n)
     * 
     * @param element the element to remove
     * @param X the set to remove from
     * @return set_t* a new set with the element removed.
     */
    set_t* remove_element(int element, set_t* X);

    /**
     * @brief Gets the subset specified by the supplied mask from the supplied set.
     * Only works for sets of sizes less than or equal to 64.
     * 
     * Complexity: O(# of 1s in the mask)
     * 
     * @param X the set to get the subset from.
     * @param mask the mask to apply on the set.
     * @return set_t* the subset extracted from the set.
     */
    set_t* get_subset(set_t* X, long long mask);

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