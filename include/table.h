
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
    table_t* make_table(int32_t n, int32_t m);

    /**
     * @brief Prints the input table in a formated way.
     *
     * @param [in] A pointer to the table_t to print.
     */
    void print_table(table_t* table);

    /**
     * @brief Frees the input table from memory.
     *
     * @param [in] A pointer to the table_t that you need to free.
     */
    void free_table(table_t* t);

#endif
