
#include <stdlib.h>

#include "pair.h"

pair_t* make_pair(void* first, void* second) {

    pair_t* p = (pair_t*) malloc(sizeof(pair_t));

    p->first = first;
    p->second = second;

    return p;

}