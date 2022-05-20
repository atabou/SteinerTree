
#ifndef CUDASET_H

    #define CUDASET_H

    #include "set.h"

    /** 
     * @brief Renames set_t to cudaset_t to show that a set is stored on the GPU. 
     */
    typedef set_t cudaset_t;

    /** 
     * @brief Copies a given set_t to the GPU, and returns a pointer to it.
     *
     * @param [in] set A pointer to the set_t to copy to the GPU.
     * @return A pointer to cudaset_t on the GPU.
     */
    cudaset_t* copy_cudaset(set_t* set);

    /**
     * @brief Frees a given cudaset_t from the GPU memory.
     *
     * @param [in] set A pointer to a cudaset_t on the GPU.
     */
    void free_cudaset(cudaset_t* set);

#endif
