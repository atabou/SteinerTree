/** 
 * \addtogroup Table
 * @{ */


#ifndef TABLE_H

    #define TABLE_H

    #include <stdint.h>

    namespace table {

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
         * @brief Prints the input table in a formated way.
         *
         * @param [in] table_t* A pointer to the table_t to print.
         */
        __device__ __host__ void print_table(table_t* table);


        /**
         * @brief Frees the input table from memory.
         *
         * @param [in] table_t* A pointer to the table_t that you need to free.
         */
        __host__ void free_table(table_t* t);

    }

    


    /********************************************************************************
    *********************************************************************************
    ****************************** cudatable functions ******************************
    *********************************************************************************
    ********************************************************************************/

    namespace cudatable {

        typedef table::table_t table_t;

        /**
         * @brief Creates a cudatable_t on the GPU and returns a pointer to it.
         *
         * @param [in] n The number of elements in one column.
         * @param [in] m The number of elements in one row.
         * @return A pointer to the created cudatable_t.
         */
        __host__ table_t* make_cudatable(int32_t n, int32_t m);
        

        /**
         * @brief Copies the the table from the GPU to the CPU.
         * 
         * @param [out] table A pointer to a pointer which will contain the copied table from the GPU.
         * @param [in] table_d A pointer to a table on the GPU.
         */
        __host__ void get_table_from_gpu(table::table_t** table, table_t* table_d);
        

        /**
         * @brief Creates a new cudatable_t on the GPU, and copies the provided table to it.
         *
         * @param [in] table A pointer to a table_t on the CPU.
         * @return A pointed to a newly created and filled cudatable_t on the GPU.
         */
        __host__ table_t* copy_cudatable(table::table_t* table);
        

        /**
         * @brief Frees a given cudatable_t from the GPU memory.
         *
         * @param [in] table A pointer to a table_t to free on the GPU.
         */
        __host__ void free_cudatable(table_t* table);        

    }


#endif
/**@}*/
