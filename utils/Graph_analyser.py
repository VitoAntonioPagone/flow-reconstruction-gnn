import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def load_graph(file_path):
    graph = torch.load(file_path)
    return graph

def load_labels(file_path):
    labels = torch.load(file_path)
    return labels

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

def plot_graph(graph):
    g = to_networkx(graph)
    nx.draw(g, with_labels=True)
    plt.show()

# Load graph from file and print its info
graph_file_path = "../dataset_graph/training/train_graphs/graph_0_label.pt"
#labels_file_path = "../dataset_graph/training/train_graphs/graph_0_label.pt"

graph = load_graph(graph_file_path)
#labels = load_labels(labels_file_path)

print_graph_info(graph)
#print_graph_info(labels)