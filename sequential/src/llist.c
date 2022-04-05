
#include <stdlib.h>

#include "llist.h"

llist* llist_add(llist* lst, int v, int w) {

    llist* l = (llist*) malloc(sizeof(llist));

    l->data = v;
    l->weight = w;
    l->next = lst;

    return l;

}

void destroy_llist(llist* lst) {

    if(lst != NULL) {

        while (lst->next != NULL) {
            
            llist* del = lst->next;
            lst->next = lst->next->next;

            del->next = NULL;
            free(del);

        }

        free(lst);
        lst->next = NULL;

    }
    

}