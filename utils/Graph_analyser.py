import torch
from torch_geometric.data import Data

def load_graph(file_path):
    graph = torch.load(file_path)
    return graph

def print_graph_info(graph):
    print("Graph Information:")
    print("------------------")
    print("Number of Nodes:", graph.num_nodes)
    print("Number of Edges:", graph.num_edges)
    print("Number of Node Features:", graph.num_node_features)
    print("Number of Edge Features:", graph.num_edge_features)
    print("Is Directed:", graph.is_directed())
    print("Contains Isolated Nodes:", graph.contains_isolated_nodes())
    print("Contains Self-loops:", graph.contains_self_loops())
    print("Is Undirected:", graph.is_undirected())

# Load graph from file and print its info
file_path = "../dataset_graph/train_graphs/graph_0.pt"  # replace with your specific file path
graph = load_graph(file_path)
print_graph_info(graph)
