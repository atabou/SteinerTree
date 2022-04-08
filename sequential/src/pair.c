
#include <stdlib.h>

#include "pair.h"

pair* make_pair(void* first, void* second) {

    pair* p = (pair*) malloc(sizeof(pair));

    p->first = first;
    p->second = second;

    return p;

}