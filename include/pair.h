
#ifndef PAIR_H

    #define PAIR_H

    typedef struct pair_t pair_t;

    struct pair_t {

        void* first;
        void* second;

    };

    /**
     * @brief Creates a new pair_t.
     * 
     * @param first a void pointer to the first data point to include in the pair_t.
     * @param second a void pointer to the second data point to include in the pair_t.
     * @return pair_t* a pointer to the constructed pair_t.
     */
    pair_t* make_pair(void* first, void* second);

#endif