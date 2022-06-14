from pysteiner import pygraph
from pysteiner import pyquery
from pysteiner import pysteiner

if __name__ == "__main__":
    
    graph = pygraph.graph()
    
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)
    pygraph.insert_vertex(graph)

    pygraph.insert_edge(graph, 0, 1, 1)
    pygraph.insert_edge(graph, 0, 1, 1)
    pygraph.insert_edge(graph, 0, 2, 1)
    pygraph.insert_edge(graph, 0, 3, 1)
    pygraph.insert_edge(graph, 0, 4, 1)
    pygraph.insert_edge(graph, 1, 5, 1)
    pygraph.insert_edge(graph, 1, 6, 1)
    pygraph.insert_edge(graph, 1, 8, 1)
    pygraph.insert_edge(graph, 2, 5, 1)
    pygraph.insert_edge(graph, 2, 7, 1)
    pygraph.insert_edge(graph, 3, 7, 1)
    pygraph.insert_edge(graph, 3, 9, 1)
    pygraph.insert_edge(graph, 4, 6, 1)
    pygraph.insert_edge(graph, 4, 9, 1)

    pygraph.insert_edge(graph, 1, 0, 1)
    pygraph.insert_edge(graph, 1, 0, 1)
    pygraph.insert_edge(graph, 2, 0, 1)
    pygraph.insert_edge(graph, 3, 0, 1)
    pygraph.insert_edge(graph, 4, 0, 1)
    pygraph.insert_edge(graph, 5, 1, 1)
    pygraph.insert_edge(graph, 6, 1, 1)
    pygraph.insert_edge(graph, 8, 1, 1)
    pygraph.insert_edge(graph, 5, 2, 1)
    pygraph.insert_edge(graph, 7, 2, 1)
    pygraph.insert_edge(graph, 7, 3, 1)
    pygraph.insert_edge(graph, 9, 3, 1)
    pygraph.insert_edge(graph, 6, 4, 1)
    pygraph.insert_edge(graph, 9, 4, 1)

    query = pyquery.query()

    pyquery.insert(query, 0)
    pyquery.insert(query, 5)
    pyquery.insert(query, 6)
    pyquery.insert(query, 7)
    pyquery.insert(query, 8)
    pyquery.insert(query, 9)

    result = pysteiner.steiner(graph, query, True)

    cost = pysteiner.cost(result)

    subgraph = pysteiner.subgraph(result)

    pygraph.to_graphviz("test.dot", subgraph)







