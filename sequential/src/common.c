
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

void print_bits(long long number, int num_bits) {

    long long bit = 1ll << (num_bits - 1);

    while(bit != 0) {

        if(number & bit) {
            printf("1");
        } else{
            printf("0");
        }

        bit = bit >> 1;

    }


}

void print_table(int** table, int n, int m) {

    printf("\n\033[0;32m   |");

    for(int i=0; i<m; i++) {
        printf("%2d|", i);
    }

    printf("\n");
    for(int i=0; i<m; i++) {
        printf("+--");
    }
    printf("+--+\033[0m\n");

    for(int i=0; i<n; i++) {
        printf("\033[0;32m %2d|\033[0m", i);
        for(int j=0; j<m; j++) {

            if(table[i][j] == -1) {
                printf("\033[0;31m%2d\033[0m|", table[i][j]);
            } else {
                printf("%2d|", table[i][j]);
            }
            
        }
        printf("\n\033[0;32m+--+\033[0m");
        for(int j=0; j<m; j++) {
            printf("--+");
        }
        printf("\n");
    }
    printf("\n");

}

void free_table(int** table, int n, int m) {

    for(int i=0; i<n; i++) {
        free(table[i]);
        table[i] = NULL;
    }

    free(table);

}