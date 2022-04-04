
#ifndef SET_H

    #define SET_H

    /**
     * Calculates the powerset of an input set.
     * Complexity: O(n*2^n) (can be improved to O(2^n)).
     * The first number of each subset contains the length of this subset.
     * 
     * @param set 
     * @param n 
     * @return int** 
     */
    int** powerset(int* set, int n);

#endif