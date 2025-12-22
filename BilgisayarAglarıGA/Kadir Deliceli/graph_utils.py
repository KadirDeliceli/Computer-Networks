import random
import networkx as nx
import pandas as pd

# csv dosyalarını okur ve ona göre düğümleri ve özellikleri oluşturur
# düğümler arası kenarları oluşturur ve özelliklerini atar
def create_graph_from_csv(edge_file, demand_file, node_file):
    """
    CSV dosyalarından graf oluşturur.
    """
    edges_df = pd.read_csv(edge_file, sep=';', decimal=',')
    nodes_df = pd.read_csv(node_file, sep=';', decimal=',')
    demands_df = pd.read_csv(demand_file, sep=';', decimal=',')

    # Sütun isimlerindeki boşlukları temizle
    nodes_df.columns = nodes_df.columns.str.strip()
    edges_df.columns = edges_df.columns.str.strip()
    demands_df.columns = demands_df.columns.str.strip()

    # Boş graf olşur burda
    G = nx.Graph()

    # Düğümleri ekle ve özelliklerini atamak için
    for _, row in nodes_df.iterrows():
        node_id = int(row['node_id'])
        G.add_node(node_id)
        G.nodes[node_id]["processing_delay"] = float(row['s_ms'])
        G.nodes[node_id]["reliability"] = float(row['r_node'])

    # Kenarları ekle ve özelliklerini atamak için yaptım
    for _, row in edges_df.iterrows():
        u = int(row['src'])
        v = int(row['dst'])
        G.add_edge(u, v)

        G.edges[u, v]["bandwidth"] = float(row['capacity_mbps'])
        G.edges[u, v]["delay"] = float(row['delay_ms'])
        G.edges[u, v]["reliability"] = float(row['r_link'])



    return G


def create_random_graph(num_nodes=250, p=0.4, edge_file=None, demand_file=None, node_file=None):
    """
    CSV dosyaları verilmişse onlardan, yoksa rastgele graf oluşturur.
    Bu şekilde eski kodunla uyumlu kalır.
    """
    # Eğer CSV dosyaları verilmişse, onlardan oku


    ## csv dosyaları verilmişse
    if edge_file and node_file:
        return create_graph_from_csv(edge_file, demand_file, node_file)

    # csv dosyları Yoksa eski random yöntemi kullan
    G = nx.erdos_renyi_graph(num_nodes, p)

    for n in G.nodes():
        G.nodes[n]["processing_delay"] = random.uniform(0.5, 2.0)
        G.nodes[n]["reliability"] = random.uniform(0.95, 0.999)

    for u, v in G.edges():
        # Bant genişliği: 100 Mbps - 1000 Mbps (rastgele)
        G.edges[u, v]["bandwidth"] = random.uniform(100.0, 1000.0)
        # Gecikme: 3 ms - 15 ms (rastgele)
        G.edges[u, v]["delay"] = random.uniform(3.0, 15.0)
        # Bağlantı güvenilirliği: 0.95 - 0.999 (rastgele)
        G.edges[u, v]["reliability"] = random.uniform(0.95, 0.999)

    return G


# Aralarında yol olmayan iki node arasındaki değerleri güncellemek istersek yada
# arassında yol olmayan başlangıç ve bitiş düğümleri varsa ikisini bağladıktan sonra parametrelerini vermek için kullanıyoruz
def assign_random_edge_attributes(G, u, v):
    """
    Verilen (u, v) kenarı için rastgele QoS (Quality of Service) parametreleri atar.

    Parametreler:
    - Bant genişliği (Bandwidth): 100-1000 Mbps arası
    - Gecikme (Delay): 3-15 ms arası
    - Güvenilirlik (Reliability): 0.95-0.999 arası
    """
    G.edges[u, v]["bandwidth"] = random.uniform(100.0, 1000.0)

    G.edges[u, v]["delay"] = random.uniform(3.0, 15.0)

    G.edges[u, v]["reliability"] = random.uniform(0.95, 0.999)

