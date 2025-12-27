import random
import networkx as nx
import pandas as pd

def create_graph_from_csv(edge_file, demand_file, node_file):
    edges_df = pd.read_csv(edge_file, sep=';', decimal=',')
    nodes_df = pd.read_csv(node_file, sep=';', decimal=',')
    demands_df = pd.read_csv(demand_file, sep=';', decimal=',')


    nodes_df.columns = nodes_df.columns.str.strip()
    edges_df.columns = edges_df.columns.str.strip()
    demands_df.columns = demands_df.columns.str.strip()


    G = nx.Graph()


    for _, row in nodes_df.iterrows():
        node_id = int(row['node_id'])
        G.add_node(node_id)
        G.nodes[node_id]["processing_delay"] = float(row['s_ms'])
        G.nodes[node_id]["reliability"] = float(row['r_node'])


    for _, row in edges_df.iterrows():
        u = int(row['src'])
        v = int(row['dst'])
        G.add_edge(u, v)

        G.edges[u, v]["bandwidth"] = float(row['capacity_mbps'])
        G.edges[u, v]["delay"] = float(row['delay_ms'])
        G.edges[u, v]["reliability"] = float(row['r_link'])



    return G


def create_random_graph(num_nodes=250, p=0.4, edge_file=None, demand_file=None, node_file=None):

    if edge_file and node_file:
        return create_graph_from_csv(edge_file, demand_file, node_file)

    # csv bulamazsa random oluşturacak
    G = nx.erdos_renyi_graph(num_nodes, p)

    for n in G.nodes():
        G.nodes[n]["processing_delay"] = random.uniform(0.5, 2.0)
        G.nodes[n]["reliability"] = random.uniform(0.95, 0.999)

    for u, v in G.edges():
        G.edges[u, v]["bandwidth"] = random.uniform(100.0, 1000.0)
        G.edges[u, v]["delay"] = random.uniform(3.0, 15.0)
        G.edges[u, v]["reliability"] = random.uniform(0.95, 0.999)

    return G


# arassında yol olmayan başlangıç ve bitiş düğümleri varsa ikisini bağladıktan sonra parametrelerini vermek için kullanıyoruz
def assign_random_edge_attributes(G, u, v):
    G.edges[u, v]["bandwidth"] = random.uniform(100.0, 1000.0)
    G.edges[u, v]["delay"] = random.uniform(3.0, 15.0)
    G.edges[u, v]["reliability"] = random.uniform(0.95, 0.999)

