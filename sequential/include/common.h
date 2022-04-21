
#ifndef COMMON_H

    #define COMMON_H

    #include "graph.h"
    #include "set.h"
    
    int load_gr_file(char* filename, graph** G, set_t** T);

    /**
     * @brief Given a pointer to a 64-bit bitmask, updates the content of the bitmask to the next combination n choose k,
     * and then returns a boolean that represents if this is the last possible combination from n choose k.
     * 
     * Complexity: O(1)
     * Complexity from a mask of 0 to the last combination: O(n choose k) 
     * 
     * @param n The number of possible elements to choose from in the combination.
     * @param k The number of elements to choose.
     * @param mask A pointer to the current bitmask to update.
     * @return int 
     */
    int next_combination(int n, int k, long long* mask);
    
    /**
     * @brief Prints the bits in the bitmask in a formatted way.
     * 
     * @param bitmask The bitmask to print.
     * @param num_bits the number of elements from the bitmask to print.
     */
    void print_bits(long long bitmask, int num_bits);

    /**
     * @brief Prints a table of integers in a formatted way.
     * 
     * @param table a 2D array of integers.
     * @param n the number of rows in the table.
     * @param m the number of columns in the table.
     */
    void print_table(int* table, int n, int m);

    /**
     * @brief Returns the maximum between the two input integer values.
     * 
     * @param x an integer.
     * @param y an integer.
     * @return int the maximum between x and y.
     */
    int max(int x, int y);

    /**
     * @brief Returns the minimum between the two input integer values.
     * 
     * @param x an integer.
     * @param y an integer.
     * @return int the minimum between x and y.
     */
    int min(int x, int y);

#endif