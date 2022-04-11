
#ifndef COMMON_H

    #define COMMON_H

    // Return the ith combination of size r for masks of size n.
    long long combination(int n, int r, int k);

    void print_table(int** table, int n, int m);
    void print_bits(long long number, int num_bits);

    void free_table(int** table, int n, int m);

#endif