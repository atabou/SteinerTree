/** 
 * \addtogroup Table
 * @{ */


#ifndef TABLE_H

    #define TABLE_H

    #include <stdint.h>

    namespace table {
       
        template<class T> struct table_t {

            T* vals; /** A 1D array that contains the n x m table. */

            int32_t  n; /** The number of elements in one column. */
            int32_t  m; /** The number of elements in one row. */

        };

        /**
         * @brief Creates a new n x m table.
         *
         * @param [in] n The number of elements in one column.
         * @param [in] m The number of elements in one row.
         * @return A pointer to the newly created table_t.
         */
        template<class T>
        void make(table_t<T>** table, int32_t n, int32_t m);


        /**
         * @brief Prints the input table in a formated way.
         *
         * @param [in] table_t* A pointer to the table_t to print.
         */
        template<class T>
        void print(table_t<T>* table);


        /**
         * @brief Frees the input table from memory.
         *
         * @param [in] table_t* A pointer to the table_t that you need to free.
         */ 
        template<class T>
        void destroy(table_t<T>* t);

    }

    


    /********************************************************************************
    *********************************************************************************
    ****************************** cudatable functions ******************************
    *********************************************************************************
    ********************************************************************************/

    namespace cudatable {

        template <class T> struct table_t {
            
            table::table_t<T>* table;
            
            int32_t n;
            int32_t m;

        };

        /**
         * @brief Creates a cudatable_t on the GPU and returns a pointer to it.
         *
         * @param [in] n The number of elements in one column.
         * @param [in] m The number of elements in one row.
         * @return A pointer to the created cudatable_t.
         */
        template<class T>
        void make(table_t<T>** table, int32_t n, int32_t m);
        

        /**
         * @brief Copies the the table from the GPU to the CPU.
         * 
         * @param [out] table A pointer to a pointer which will contain the copied table from the GPU.
         * @param [in] table_d A pointer to a table on the GPU.
         */
        template<class T>
        void transfer_from_gpu(table::table_t<T>** table, table_t<T>* table_d);
        

        /**
         * @brief Creates a new cudatable_t on the GPU, and copies the provided table to it.
         *
         * @param [in] table A pointer to a table_t on the CPU.
         * @return A pointed to a newly created and filled cudatable_t on the GPU.
         */
        template<class T>
        void transfer_to_gpu(table_t<T>** table_d, table::table_t<T>* table);
        

        /**
         * @brief Frees a given cudatable_t from the GPU memory.
         *
         * @param [in] table A pointer to a table_t to free on the GPU.
         */
        template<class T>
        void destroy(table_t<T>* table);        

    }

#endif
/**@}*/
