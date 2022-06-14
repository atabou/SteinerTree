
#include <pybind11/pybind11.h>

#include "util.hpp"
clock_t CLOCKMACRO;
#include "table.hpp"
#include "graph.hpp"
#include "query.hpp"
#include "shortestpath.hpp"
#include "steiner.hpp"


struct graph_wrapper {

    graph::graph_t* blob;

    graph_wrapper() {

        graph::graph_t* graph;
        
        graph::make(&graph);

        this->blob = graph;

    }

};


struct query_wrapper {

    query::query_t* blob;

    query_wrapper() {
 
        query::query_t* query;

        query::make(&query);
        
        this->blob = query;

    }

};


struct steiner_result_wrapper {

    steiner::result_t* blob;

};


PYBIND11_MODULE(pysteiner, m) {

    m.attr("__version__") = "0.0.1";

    pybind11::module pygraph = m.def_submodule("pygraph", "Module that contains the necessary functions to create and construct a graph.");


    pybind11::class_<graph_wrapper>(pygraph, "graph")
        .def(pybind11::init<>());


    pygraph.def(
        "insert_vertex", 
        [](graph_wrapper wrapper) {
            graph::graph_t* graph = wrapper.blob;
            int id = graph::insert_vertex(graph);
            return id;
        }
    );


    pygraph.def(
        "insert_edge",
        [](graph_wrapper wrapper, int32_t src, int32_t dst, float wgt) {
            graph::graph_t* graph = wrapper.blob;
            graph::insert_edge(graph, src, dst, wgt);
        }
    );

    
    pygraph.def(
        "weight",
        [](graph_wrapper wrapper, int32_t src, int32_t dst) {
            graph::graph_t* graph = wrapper.blob;
            return graph::weight(graph, src, dst);
        }
    );

    
    pygraph.def(
        "to_graphviz",
        [](char* filename, graph_wrapper wrapper) {
            graph::graph_t* graph = wrapper.blob;
            graph::to_graphviz(graph, filename);
        }
    );

    
    pygraph.def(
        "destroy",
        [](graph_wrapper wrapper) {
            graph::graph_t* graph = wrapper.blob;
            graph::destroy(graph);
        }
    );

        
    pybind11::module pyquery = m.def_submodule("pyquery", "Module that defines the query class and functions related to it.");
      
    
    pybind11::class_<query_wrapper>(pyquery, "query")
        .def(pybind11::init<>());
 

    pyquery.def(
        "insert",
        [](query_wrapper wrapper, int32_t element) {
            query::query_t* query = wrapper.blob;
            query::insert(query, element);
        }
    );

    
    pyquery.def(
        "destroy",
        [](query_wrapper wrapper) {
            query::query_t* query = wrapper.blob;
            query::destroy(query);
            return pybind11::none();
        }
    );

   
    pybind11::module pyst = m.def_submodule("pysteiner", "Contains the function to compute the minimum steiner tree.");


    pybind11::class_<steiner_result_wrapper>(pygraph, "steiner_result");


    pyst.def(
        "steiner",
        [](graph_wrapper gwrapper, query_wrapper qwrapper, bool run_on_gpu) {
            
            graph::graph_t* graph = gwrapper.blob;
            query::query_t* terms = qwrapper.blob;

            table::table_t< float >* dists = NULL;
            table::table_t<int32_t>* preds = NULL;

            table::make(&dists, graph->vrt, graph->vrt); 
            table::make(&preds, graph->vrt, graph->vrt);

            apsp_gpu_graph(graph, dists, preds);

            steiner::result_t* opt = NULL;

            if(run_on_gpu) {
                
                cudagraph::graph_t*        graph_d = NULL; 
                cudatable::table_t<float>* dists_d = NULL;
                cudaquery::query_t*        terms_d = NULL;

                cudagraph::transfer_to_gpu(&graph_d, graph);
                cudatable::transfer_to_gpu(&dists_d, dists);
                cudaquery::transfer_to_gpu(&terms_d, terms);

                TIME(steiner::fill(graph_d, terms_d, dists_d, &opt), "\tTable fill:");
                TIME(steiner::backtrack(terms, opt), "\tBacktracking:");
                TIME(steiner::branch_and_clean(preds, opt), "\tBranch and clean:");
                TIME(steiner::build_subgraph(graph, opt), "\tBuild subgraph:");

                cudaquery::destroy(terms_d);
                cudatable::destroy(dists_d);
                cudagraph::destroy(graph_d);

            } else {

                steiner::fill(graph, terms, dists, &opt);
                steiner::backtrack(terms, opt);
                steiner::branch_and_clean(preds, opt);
                steiner::build_subgraph(graph, opt);

            }

            table::destroy(dists);
            table::destroy(preds);
        
            steiner_result_wrapper wrapper;

            wrapper.blob = opt;

            return wrapper;

        }
    );


    pyst.def(
        "cost",
        [](steiner_result_wrapper wrapper) {

            steiner::result_t* opt = wrapper.blob;

            return opt->cost;

        }
    );


    pyst.def(
        "subgraph",
        [](steiner_result_wrapper wrapper) {

            steiner::result_t* opt = wrapper.blob;

            graph::graph_t* subgraph = opt->subgraph;

            graph_wrapper result;

            result.blob = subgraph;

            return result;

        }
    );

}
