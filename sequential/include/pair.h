
#ifndef PAIR_H

    #define PAIR_H

    typedef struct pair pair;

    struct pair {

        void* first;
        void* second;

    };

    pair* make_pair(void* first, void* second);

#endif