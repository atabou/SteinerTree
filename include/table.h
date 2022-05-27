/** 
 * \addtogroup Table
 * @{ */


#ifndef TABLE_H

    #define TABLE_H

    #include <stdint.h>


    typedef struct table_t {

        float* vals; /** A 1D array that contains the n x m table. */

        int32_t  n; /** The number of elements in one column. */
        int32_t  m; /** The number of elements in one row. */

    } table_t;


    /**
     * @brief Creates a new n x m table.
     *
     * @param [in] n The number of elements in one column.
     * @param [in] m The number of elements in one row.
     * @return A pointer to the newly created table_t.
     */
    __host__ table_t* make_table(int32_t n, int32_t m);


    /**
     * @brief Frees the input table from memory.
     *
     * @param [in] table_t* A pointer to the table_t that you need to free.
     */
    __host__ void free_table(table_t* t);


    /********************************************************************************
    *********************************************************************************
    ****************************** cudatable functions ******************************
    *********************************************************************************
    ********************************************************************************/


    typedef table_t cudatable_t; /** @brief Renamed table_t to cudatable_t to show that the cudatable_t is actually stored in the GPU. */


    /**
     * @brief Creates a cudatable_t on the GPU and returns a pointer to it.
     *
     * @param [in] n The number of elements in one column.
     * @param [in] m The number of elements in one row.
     * @return A pointer to the created cudatable_t.
     */
    __host__ cudatable_t* make_cudatable(int32_t n, int32_t m);
    

    /**
     * @brief Copies the the table from the GPU to the CPU.
     * 
     * @param [in] table A pointer to a table on the GPU.
     * @return A pointer to the copied table on the CPU.
     */
    __host__ table_t* get_table_from_gpu(cudatable_t* table);
    

    /**
     * @brief Creates a new cudatable_t on the GPU, and copies the provided table to it.
     *
     * @param [in] table A pointer to a table_t on the CPU.
     * @return A pointed to a newly created and filled cudatable_t on the GPU.
     */
    __host__ cudatable_t* copy_cudatable(table_t* table);
    

    /**
     * @brief Frees a given cudatable_t from the GPU memory.
     *
     * @param [in] table A pointer to a cudatable_t to free on the GPU.
     */
    __host__ void free_cudatable(cudatable_t* table);


    /********************************************************************************
    *********************************************************************************
    ******************************* global functions ********************************
    *********************************************************************************
    ********************************************************************************/


    /**
     * @brief Prints the input table in a formated way.
     *
     * @param [in] table_t* A pointer to the table_t to print.
     */
    __device__ __host__ void print_table(table_t* table);


#endif
/**@}*/
