/** 
 * \addtogroup CUDACombination 
 * @{ */

#ifndef COMBINATION_CUDA_H

    #define COMBINATION_CUDA_H

    #include <stdint.h>

    /**
     * @brief Prints the bits in the bitmask in a formatted way.
     * 
     * @param mask The bitmask to print.
     * @param size The number of elements from the bitmask to print.
     */    
    __device__ __host__ void print_mask(uint64_t mask, int32_t size);


    /**
     * @brief Given a pointer to a 64-bit bitmask, updates the content of the bitmask to the next combination n choose k,
     * and then returns a boolean that represents if this is the last possible combination from n choose k.
     * 
     * Complexity: O(1)
     * Complexity from a mask of 0 to the last combination: O(n choose k) 
     * 
     * @param [in] n The number of possible elements to choose from in the combination.
     * @param [in] k The number of elements to choose.
     * @param [in,out] mask A pointer to the current bitmask to update.
     * @return 0 if no next combination of size exists, 1 otherwise.
     */
    __device__ __host__ int next_combination(int32_t n, int32_t k, uint64_t* mask);


    extern __constant__ double factorial[65];
    extern double factorial_h[65];


    __host__ void initialize_factorial_table();

    
    /**
     * @brief 
     * Must have previously called initialize_factorial_table function.
     * 
     * @param n 
     * @param k 
     * @param i 
     * @return __device__ 
     */
    __device__ uint64_t ith_combination(int32_t n, int32_t k, uint64_t i);


    /**
     * @brief Calculates the value of n choose r (number of combination).
     * Can only calculate with n <= 64 and r <= n.
     * Must have previously called initialize_factorial_table function.
     * 
     * @param n The number of elements to choose from.
     * @param r The number of elements to choose.
     * @return The number of possible combinations.
     */
    __device__ __host__ uint64_t nCr(int32_t n, int32_t r);
    

#endif
/**@}*/
