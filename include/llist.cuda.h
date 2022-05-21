/** 
 * \addtogroup LinkedListCUDA
 * @{ */

#ifndef CUDALLIST_H

    #define CUDALLIST_H

    #include "llist.h"

    /**
     * @brief Renamed llist_t to cudallist_t as to make it clear that a cudallist_t is stored on the GPU.
     */
    typedef llist_t cudallist_t;

    /**
     * @brief Copies a given cudallist_t to the GPU.
     *
     * @param [in] lst A pointer to the cudallist_t.
     * @param [in] size The size of the cudallist_t to copy.
     * @return A pointer to the copied cudallist_t* stored on the GPU.
     */
    cudallist_t* copy_cudallist(llist_t* lst, int32_t size);

    /**
     * @brief Frees a given pointer to a cudallist_t from the GPU memory.
     *
     * @param [in] lst A pointer to a cudallist_t on the GPU.
     */
    void free_cudallist(cudallist_t* lst);

#endif
/**@}*/
