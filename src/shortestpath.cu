#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#include "shortestpath.h"
#include "cugraph_c/graph.h"
#include "cugraph_c/algorithms.h"

void toCOO(graph::graph_t* graph, int32_t** src, int32_t** dst, float** wgt, int32_t* nedg) {

    *nedg = 0;

    for(int32_t i=0; i<graph->vrt; i++) {
        *nedg += graph->deg[i];
    }

    *src = (int32_t*) malloc(sizeof(int32_t) * (*nedg));
    *dst = (int32_t*) malloc(sizeof(int32_t) * (*nedg));
    *wgt = ( float* ) malloc(sizeof( float ) * (*nedg));

    int32_t start = 0;

    for(int32_t i=0; i<graph->vrt; i++) {

        for(int32_t j=0; j<graph->deg[i]; j++) {
            (*src)[start + j] = i;
        }

        memcpy(&((*dst)[start]), graph->dst[i], sizeof(int32_t) * graph->deg[i]);
        memcpy(&((*wgt)[start]), graph->wgt[i], sizeof( float ) * graph->deg[i]);
        
        start += graph->deg[i];

    }

}

void destroy_gpu_graph(cugraph_graph_t* graph) {
    cugraph_sg_graph_free(graph);
}

cugraph_graph_t* create_gpu_graph(cugraph_resource_handle_t* handle, int32_t* hsrc, int32_t* hdst, float* hwgt, int32_t nedg) {

    cugraph_error_t* error = NULL;
    cugraph_error_code_t status = CUGRAPH_SUCCESS;

    // Initialize device arrays

    cugraph_type_erased_device_array_t* src;
    cugraph_type_erased_device_array_t* dst;
    cugraph_type_erased_device_array_t* wgt;

    status = cugraph_type_erased_device_array_create(handle, nedg, INT32, &src, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        exit(status);
    }

    status = cugraph_type_erased_device_array_create(handle, nedg, INT32, &dst, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        exit(status);
    }

    status = cugraph_type_erased_device_array_create(handle, nedg, FLOAT32, &wgt, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        exit(status);
    }

    // Initialize array views

    cugraph_type_erased_device_array_view_t* srcv; 
    cugraph_type_erased_device_array_view_t* dstv;
    cugraph_type_erased_device_array_view_t* wgtv;

    srcv = cugraph_type_erased_device_array_view(src);
    dstv = cugraph_type_erased_device_array_view(dst);
    wgtv = cugraph_type_erased_device_array_view(wgt);

    status = cugraph_type_erased_device_array_view_copy_from_host(handle, srcv, (byte_t*) hsrc, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        printf("%d\n", status);
        exit(status);
    }

    status = cugraph_type_erased_device_array_view_copy_from_host(handle, dstv, (byte_t*) hdst, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        exit(status);
    }

   status = cugraph_type_erased_device_array_view_copy_from_host(handle, wgtv, (byte_t*) hwgt, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        exit(status);
    }

    // Initialize graph

    cugraph_graph_properties_t properties;

    properties.is_symmetric  = FALSE;
    properties.is_multigraph = FALSE;

    cugraph_graph_t* graph;

    status = cugraph_sg_graph_create(handle, &properties, srcv, dstv, wgtv, FALSE, FALSE, FALSE, &graph, &error);

    if(status != CUGRAPH_SUCCESS) {
        printf("%s\n", cugraph_error_message(error));
        exit(status);
    }

    cugraph_type_erased_device_array_view_free(wgtv);
    cugraph_type_erased_device_array_view_free(dstv);
    cugraph_type_erased_device_array_view_free(srcv);
    cugraph_type_erased_device_array_free(src);
    cugraph_type_erased_device_array_free(dst);
    cugraph_type_erased_device_array_free(wgt);


    return graph;

}

void apsp_gpu_graph(graph::graph_t* graph, table::table_t* distances, table::table_t* predecessors) {

    cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);

    int32_t  nedg = 0;

    int32_t* src = NULL;
    int32_t* dst = NULL;
    float*   wgt = NULL;
 
    /* clock_t c = clock(); */
    toCOO(graph, &src, &dst, &wgt, &nedg);
    /* printf("\t\t- Convert graph to COO format: %fs\n", (double) (clock() - c) / CLOCKS_PER_SEC); */

    cugraph_graph_t* cugraph = create_gpu_graph(handle, src, dst, wgt, nedg);
    cugraph_paths_result_t* result = NULL;
    cugraph_error_t* error = NULL;
    cugraph_error_code_t status = CUGRAPH_SUCCESS;

    /* double duration = 0; */

    for(int32_t source=0; source < distances->n; source++) {
 
        /* c = clock(); */
        status = cugraph_sssp(handle, cugraph, source, FLT_MAX, TRUE, FALSE, &result, &error);
        /* duration = duration + (double) (clock() - c) / CLOCKS_PER_SEC; */


        if(status != CUGRAPH_SUCCESS) {
            printf("%s\n", cugraph_error_message(error));
            exit(status);
        }

        cugraph_type_erased_device_array_view_t* distv = cugraph_paths_result_get_distances(result);

        status = cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*) &(distances->vals[source * distances->m]), distv, &error);

        if(status != CUGRAPH_SUCCESS) {
            printf("%s\n", cugraph_error_message(error));
            exit(status);
        }

        cugraph_type_erased_device_array_view_free(distv);

        cugraph_type_erased_device_array_view_t* predv = cugraph_paths_result_get_predecessors(result);

        status = cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*) &(predecessors->vals[source * distances->m]), predv, &error);

        if(status != CUGRAPH_SUCCESS) {
            printf("%s\n", cugraph_error_message(error));
            exit(status);
        }

        cugraph_type_erased_device_array_view_free(predv);
        
        cugraph_paths_result_free(result);
        result = NULL;

    }

    /* printf("\t\t- SSSP: %fs (total) - %fs (avg over %d calls)\n", duration, duration / graph->vrt, graph->vrt); */

    cugraph_free_resource_handle(handle);

    destroy_gpu_graph(cugraph);
    
    free(src);
    free(dst);
    free(wgt);

}

