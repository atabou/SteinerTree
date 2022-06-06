/**
 * \addtogroup Set
 * @{ */

#ifndef SET_H

    #define SET_H

    #include <stdint.h>

    namespace query {


        typedef struct query_t {

            int32_t* vals;
            int32_t  size;

        } query_t;


        /**
         * @brief Creates a new empty set.
         *
         * O(1)
         *
         * @return set_t* the created set.
         */
        __host__ void make(query_t** q);


        /**
         * @brief Inserts a new integer in the set. If the element already exists nothing is done.
         *
         * Complexity: O(n)
         *
         * @param [in] set A pointer to the set to insert in.
         * @param [in] x The element to insert in the set.
         */
        __host__ void insert(query_t* set, int32_t x);


        /**
         * @brief Checks wether the specified element exists in the set and inside the specified mask.
         *
         * Complexity: O(n)
         *
         * @param [in] element The element to search for.
         * @param [in] set A pointer to the set_t to search in.
         * @param [in] mask A mask over which the search is considered.
         * @return int
         */
        int element_exists(int32_t element, query_t* set, uint64_t mask);


        /**
         * @brief Returns the position of the specified element in the set. If the element does not exist -1 is returned.
         *
         * @param [in] set A pointer to a set_t to search in.
         * @param [in] element The element to search for.
         */
        int32_t find_position(query_t* set, int32_t element);


        /**
         * @brief Prints the set in a formatted way.
         *
         * @param [in] X A pointer to a set_t.
         */
        __device__ __host__ void print(query_t* X);


        /**
         * @brief Destroys and frees the set.
         *
         * @param [in] set A pointer to a set_t.
         */
        __host__ void destroy(query_t* set);

    }

    namespace cudaquery {


        typedef query::query_t query_t; /** @brief Renames set_t to cudaset_t to make it clear that a set_t is stored on the GPU. */


        /**
         * @brief Copies a given set_t to the GPU, and returns a pointer to it.
         *
         * @param [in] set A pointer to the set_t to copy to the GPU.
         * @return A pointer to cudaset_t on the GPU.
         */
        __host__ void transfer_to_gpu(query_t** set_d, query::query_t* set);

        __host__ void transfer_from_gpu(query::query_t** query, query_t* query_d);

        /**
         * @brief Frees a given cudaset_t from the GPU memory.
         *
         * @param [in] set A pointer to a cudaset_t on the GPU.
         */
        __host__ void destroy(query_t* set);


    }   

#endif
/**@}*/
