
#ifndef COMBINATION_CUDA_H

    #define COMBINATION_CUDA_H

    #include <stdint.h>

    /**
     * @brief Prints the specified mask in binary form.
     * 
     * @param [in] mask the mask to print.
     * @param [in] size the size of the mask.
     */
    __device__ __host__ void print_mask(uint64_t mask, int32_t size);

    /**
     * @brief Returns the number of possible combination of size r from n.
     * 
     * @param [in] n The number of elements to chose from.
     * @param [in] r The number of elements to choose.
     * @return uint64_t The number of possible combinations.
     */
    __device__ __host__ uint64_t nCr(uint64_t n, uint64_t r);

    /**
     * @brief Returns the ith combination of size k in the lexicographic order of combinations.
     * 
     * @param [in] n The number of elements to choose from.
     * @param [in] k The number of elements to choose.
     * @param [in] i The index of the wanted combination.
     * @return The ith combination of size k from the total n.
     */
    __device__ uint64_t ith_combination(uint64_t n, uint64_t k, uint64_t i);

    /**
     * @brief Returns the ith subset of elements from the given print_mask.
     *
     * @param [in] mask The mask to subset over.
     * @param [in] i The index of the wanted subset.
     * @return The ith subset of the provied mask.
     */
    __device__ uint64_t ith_subset(uint64_t mask, uint64_t i);

    /**
     * @brief Returns the next combination of size k given the previous combination.
     *
     * @param [in] n The number of elements to choose from.
     * @param [in] k The number of elements to choose.
     * @param [in] mask The previous combination.
     * @return The next combination in the lexicographic order.
     */
    __device__ int gpu_next_combination(uint32_t n, uint32_t k, uint64_t* mask);

#endif

