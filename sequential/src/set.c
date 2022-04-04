
#include "set.h"
#include "math.h"

int** powerset(int* set, int n) {

    int** pset = (int**) malloc(sizeof(int*) * (int) pow(2, n));

    int count=0;
    int buffer[n];

    for(int i=0; i<pow(2, n); i++) {

        for(int j=0; j<n; j++) {

            if(i & (1 << j) > 0) {

                buffer[count] = j;
                count++;
            }

        }

        int* subset = (int*) malloc(sizeof(int) * (count + 1));
        subset[0] = count;

        for(int j=0; j<subset[0]; j++) {
            subset[j + 1] = buffer[j];
        }

        pset[i] = subset;

        count = 0;

    }

    return pset;

}