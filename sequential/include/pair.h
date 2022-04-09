
#ifndef PAIR_H

    #define PAIR_H

    typedef struct pair pair;

    struct pair {

        void* first;
        void* second;

    };

    /**
     * @brief Creates a new pair.
     * 
     * @param first a void pointer to the first data point to include in the pair.
     * @param second a void pointer to the second data point to include in the pair.
     * @return pair* a pointer to the constructed pair.
     */
    pair* make_pair(void* first, void* second);

#endif