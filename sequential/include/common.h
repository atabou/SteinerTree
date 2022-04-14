
#ifndef COMMON_H

    #define COMMON_H

    int next_combination(int n, int k, long long* mask);
    
    void print_table(int** table, int n, int m);
    void print_bits(long long number, int num_bits);

    void free_table(int** table, int n, int m);

    int max(int x, int y);
    int min(int x, int y);

#endif