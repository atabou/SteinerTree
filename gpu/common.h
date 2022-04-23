
#ifndef COMMON_H

	#define COMMON_H

	typedef struct table_t table_t;

	struct table_t {
	
		uint32_t* vals;
		uint64_t  n;
		uint64_t  m;
	
	};

	typedef struct set_t set_t;

	struct set_t {
	
		uint32_t* vals;
		uint32_t  size;
	
	};

	__device__ __host__ uint64_t nCr(uint64_t n, uint64_t r);
	__host__ void steiner(graph* g, set_t* t);

#endif

