
#ifndef UTIL_H

    #define UTIL_H

    #include <stdint.h>

    /**
     * @brief Given a pointer to a 64-bit bitmask, updates the content of the bitmask to the next combination n choose k,
     * and then returns a boolean that represents if this is the last possible combination from n choose k.
     * 
     * Complexity: O(1)
     * Complexity from a mask of 0 to the last combination: O(n choose k) 
     * 
     * @param n The number of possible elements to choose from in the combination.
     * @param k The number of elements to choose.
     * @param mask A pointer to the current bitmask to update.
     * @return int 
     */
    int next_combination(uint32_t n, uint32_t k, uint64_t* mask);

    /**
     * @brief Prints the bits in the bitmask in a formatted way.
     * 
     * @param bitmask The bitmask to print.
     * @param num_bits the number of elements from the bitmask to print.
     */
    void print_bits(uint64_t number, uint32_t size);


#endif