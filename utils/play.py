import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, get_laplacian

def print_graph_info(graph):
    print("Graph Information:")
    print("------------------")
    
    print("Number of Nodes:", graph.num_nodes)
    print("Number of Edges:", graph.num_edges)
    print("Number of Node Features:", graph.num_node_features)
    print("Number of Edge Features:", graph.num_edge_features)
    print("Is Directed:", graph.is_directed())
    print("Contains Isolated Nodes:", graph.has_isolated_nodes())
    print("Contains Self-loops:", graph.has_self_loops())
    print("Is Undirected:", graph.is_undirected())

    # Convert the graph to a networkx graph for additional analysis
    g = to_networkx(graph, to_undirected=True)
    
    # Degree Distribution
    degrees = [g.degree(n) for n in g.nodes()]
    print("Average Degree:", np.mean(degrees))
    print("Minimum Degree:", np.min(degrees))
    print("Maximum Degree:", np.max(degrees))

    # Check if the graph is connected
    print("Is Connected:", nx.is_connected(g))

    # Get the number of connected components
    print("Number of Connected Components:", nx.number_connected_components(g))
    
    # Calculate the Laplacian regularization term
    laplacian_regularization_term = laplacian_regularization(graph)
    print("Laplacian Regularization Term:", laplacian_regularization_term)

def laplacian_regularization(graph):
    features = graph.x
    laplacian_indices, laplacian_values = get_laplacian(graph.edge_index, normalization=None)

    # Create Laplacian matrix
    num_nodes = graph.num_nodes
    laplacian = torch.sparse_coo_tensor(laplacian_indices, laplacian_values, size=(num_nodes, num_nodes))

    # Compute L * X
    LX = torch.sparse.mm(laplacian, features)
    
    # Compute X^T * (L * X)
    regularization_matrix = torch.mm(features.transpose(0, 1), LX)
    
    regularization = torch.trace(regularization_matrix)
    
    return regularization




def main():
    # Load Data object from .pt file
    data = torch.load('../dataset_graph/training_FP/test_graphs_98/cyc10_CAD615_Y4_Z0_X2_label.pt')
    print_graph_info(data)

if __name__ == "__main__":
    main()
