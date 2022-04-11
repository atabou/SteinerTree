
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
            printf("%2d|", table[i][j]);
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

typedef struct queue queue;



int binomial_coeficient(int n, int r) {

    int coef = 1;

    for(int i=0; i<r; i++) {
        coef = coef * (n-i);
        coef = coef / (i+1);
    }

    return coef;

}

long long combination(int n, int r, int k) {

    printf("%d, %d, %d\n", n, r, k);

    if (r == 0 || n == 0 || r > n || k >= binomial_coeficient(n, r)) { // O(r)
        
        return 0;

    }

    if(k == 0) {



    }

    if(k < binomial_coeficient(n-1, r-1)) {

        if(k == 0) {

            long long comb = combination(n-1, r-1, k);

            comb = comb << 1;
            comb = comb | 1ll;

            return comb;

        } else {

            long long comb = combination(n-1, r, k-1);

            printf("Here2\n");

            comb = comb << 1;

            return comb;

        }
       

    }

    // int[][] combs
    // for (int i = 1 to i <= n) 
    //     combs.add([i])
    // for (int i = 2 to i <= k) {
    //     int[][] newCombs
    //     for (int j = i to j <= n) {
    //         for (int[] comb in combs) {
    //             if (comb[comb.size()-1] < j) {
    //                 newComb = comb
    //                 newComb.add(j)
    //                 newCombs.add(newComb)
    //             }
    //         }
    //     }
    //     combs = newCombs
    // }
    // return combs

}