
#ifndef CUDATABLE_H

    #define CUDATABLE_H

    #include "table.h"

    /**
     * @brief Renamed table_t to cudatable_t to show that the cudatable_t is actually stored in the GPU.
     */
    typedef table_t cudatable_t;

    /**
     * @brief Creates a cudatable_t on the GPU and returns a pointer to it.
     *
     * @param [in] n The number of elements in one column.
     * @param [in] m The number of elements in one row.
     * @return A pointer to the created cudatable_t.
     */
    cudatable_t* make_cudatable(int32_t n, int32_t m);
    
    /**
     * @brief Creates a new cudatable_t on the GPU, and copies the provided table to it.
     *
     * @param [in] table A pointer to a table_t on the CPU.
     * @return A pointed to a newly created and filled cudatable_t on the GPU.
     */
    cudatable_t* copy_cudatable(table_t* table);
    
    /**
     * @brief Frees a given cudatable_t from the GPU memory.
     *
     * @param [in] table A pointer to a cudatable_t to free on the GPU.
     */
    void free_cudatable(cudatable_t* table);



#endif
