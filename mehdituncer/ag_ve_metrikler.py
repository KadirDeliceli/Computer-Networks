
import networkx as nx
import random
import math

def create_random_graph(num_nodes=250, p=0.4, seed=None):
    """
    Rastgele Ağ Topolojisi Oluşturucu
    
    Parametreler:
    - num_nodes (int): Düğüm sayısı (Varsayılan: 250)
    - p (float): Bağlantı olasılığı (Varsayılan: 0.4)
    - seed (int): Tekrarlanabilirlik için rastgele tohum
    
    Dönüş:
    - G (networkx.Graph): Oluşturulan ağ grafiği
    """
    if seed is not None:
        random.seed(seed)
    
    # 2.1 Ağ Topolojisi: Erdős–Rényi G(n, p) model
    G = nx.erdos_renyi_graph(n=num_nodes, p=p, seed=seed)
    
    # Gereksinim: Grafiğin bağlı (connected) olduğundan emin olunmalı
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        # Bileşenleri birbirine bağla
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i+1]))
            G.add_edge(u, v)
    
    # 2.2 Düğüm (Node) Özellikleri
    for node in G.nodes():
        # İşlem Süresi: [0.5ms - 2.0ms]
        G.nodes[node]['processing_delay'] = random.uniform(0.5, 2.0)
        # Düğüm Güvenilirliği: [0.95, 0.999]
        G.nodes[node]['reliability'] = random.uniform(0.95, 0.999)
        
    # 2.3 Bağlantı (Link) Özellikleri
    for u, v in G.edges():
        # Bant Genişliği: [100 Mbps, 1000 Mbps]
        G.edges[u, v]['bandwidth'] = random.uniform(100, 1000)
        # Gecikme: [3 ms, 15 ms]
        G.edges[u, v]['delay'] = random.uniform(3.0, 15.0)
        # Bağlantı Güvenilirliği: [0.95, 0.999]
        G.edges[u, v]['reliability'] = random.uniform(0.95, 0.999)
        
    return G

def compute_metrics(G, path):
    """
    Bir yol (path) için optimizasyon metriklerini hesaplar.
    
    Metrikler:
    1. Toplam Gecikme (Total Delay) - Minimize
    2. Toplam Güvenilirlik (Total Reliability) - Maximize
    3. Ağ Kaynak Kullanımı (Resource Usage) - Minimize
    
    Parametreler:
    - G: NetworkX grafiği
    - path: Düğüm listesi [S, n1, n2, ..., D]
    
    Dönüş:
    - dict: {'total_delay', 'total_reliability', 'resource_cost', 'reliability_cost'}
    """
    if not path or len(path) < 2:
        return {
            'total_delay': 0.0,
            'total_reliability': 0.0,
            'resource_cost': 0.0,
            'reliability_cost': float('inf')
        }

    total_delay = 0.0
    total_reliability = 1.0
    resource_cost = 0.0
    
    # Yol üzerindeki kenarları dolaş
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        if G.has_edge(u, v):
            edge_data = G.edges[u, v]
            
            # 3.1 Toplam Gecikme (Link Gecikmesi kısmı)
            total_delay += edge_data.get('delay', 0.0)
            
            # 3.2 Toplam Güvenilirlik (Link kısmı)
            total_reliability *= edge_data.get('reliability', 1.0)
            
            # 3.3 Ağ Kaynak Kullanımı
            # Maliyet = 1 Gbps / Bandwidth
            # Bandwidth Mbps cinsinden (1 Gbps = 1000 Mbps)
            bw = edge_data.get('bandwidth', 100.0)
            resource_cost += (1000.0 / bw)
            
    # Yol üzerindeki düğümleri dolaş (İşlem gecikmesi ve Node güvenilirliği)
    for i, node in enumerate(path):
        node_data = G.nodes[node]
        
        # 3.1 Toplam Gecikme (İşlem Gecikmesi kısmı)
        # Kaynak (S) ve Hedef (D) hariç ara düğümler için
        if i != 0 and i != len(path) - 1:
            total_delay += node_data.get('processing_delay', 0.0)
            
        # 3.2 Toplam Güvenilirlik (Node kısmı)
        # Tüm düğümler dahil
        total_reliability *= node_data.get('reliability', 1.0)

    # Güvenilirlik Maliyeti (Minimizasyon için dönüştürülmüş)
    # ReliabilityCost = sum(-log(link_rel)) + sum(-log(node_rel))
    # Bu matematiksel olarak -log(TotalReliability)'ye eşittir.
    reliability_cost = 0.0
    if total_reliability > 0:
        reliability_cost = -math.log(total_reliability)
    else:
        reliability_cost = float('inf')

    return {
        'total_delay': total_delay, 
        'total_reliability': total_reliability,
        'resource_cost': resource_cost,
        'reliability_cost': reliability_cost
    }

if __name__ == "__main__":
    # Test Bloğu
    print("--- Test Ediliyor ---")
    my_graph = create_random_graph(num_nodes=10, p=0.4, seed=42)
    print(f"Grafik oluşturuldu: {len(my_graph.nodes)} düğüm, {len(my_graph.edges)} bağlantı")
    
    # Basit bir yol testi (eğer varsa)
    try:
        path = nx.shortest_path(my_graph, source=0, target=9)
        print(f"Örnek Yol: {path}")
        metrics = compute_metrics(my_graph, path)
        print("Metrikler:", metrics)
    except nx.NetworkXNoPath:
        print("0 ve 9 arasında yol bulunamadı.")
