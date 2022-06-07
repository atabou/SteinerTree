
#ifndef TREE_H

    #define TREE_H

    #include <stdint.h>
    #include <vector>

    namespace tree {

        struct tree_t {

            int32_t vertex;
            std::vector<tree_t*> subtrees;

        };

        void make(tree_t** tree, int32_t vertex);

        void print(tree_t* tree);

        void destroy(tree_t* tree);

    };

#endif
