

#include <stdio.h>


#include "tree.hpp"


void tree::make(tree::tree_t** tree, int32_t vertex) {

    *tree = new tree::tree_t();

    (*tree)->vertex = vertex;
    (*tree)->subtrees = std::vector<tree::tree_t*>();

}


void tree::print(tree::tree_t* t) {

    printf("%d", t->vertex);
   

    for(int i=0; i<t->subtrees.size(); i++) {

        printf(" (");
        tree::print(t->subtrees[i]);
        printf(")");

    }


}


void tree::destroy(tree::tree_t* tree) {

    for(int i=0; i<tree->subtrees.size(); i++) {
        tree::destroy(tree->subtrees[i]);
    }

    delete tree;

}
