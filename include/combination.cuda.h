
#ifndef COMBINATION_H

#define COMBINATION_H

    #include <stdint.h>

    /**
     * @brief Prints the bits inside a mask.
     * 
     * @param mask the mask to print.
     * @param size the size of the mask.
     */
    __device__ __host__ void print_mask(uint64_t mask, uint32_t size);

    /**
     * @brief Returns the number of possible combination of size r from n.
     * 
     * @param n The number of elements to chose from.
     * @param r The number of elements to choose.
     * @return uint64_t
     */
    __device__ __host__ uint64_t nCr(uint64_t n, uint64_t r);

    /**
     * @brief Returns the ith combination of size k in the lexicographic order of combinations.
     * O(n^2)
     * 
     * @param n The number of elements to choose from.
     * @param k The number of elements to choose.
     * @param i The number of e
     * @return __device__ 
     */
    __device__ uint64_t ith_combination(uint64_t n, uint64_t k, uint64_t i);

    __device__ uint64_t ith_subset(uint64_t mask, uint64_t i);

#endif

