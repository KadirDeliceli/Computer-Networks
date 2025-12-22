import networkx as nx
import random
import json
import pickle
import os

print("BSM307 Mock Network Generator")
print("=" * 50)

class MockNetworkGenerator:
    def __init__(self, n_nodes=50):  # Test için 50, sonra 250
        self.n_nodes = n_nodes
        
    def generate(self):
        """Test ağı oluştur"""
        print(f"{self.n_nodes} düğümlü test ağı oluşturuluyor...")
        
        # Basit bir ağ oluştur
        G = nx.erdos_renyi_graph(self.n_nodes, 0.3)
        
        # Bağlı değilse bağla
        if not nx.is_connected(G):
            print(" Ağ bağlı değil, bağlanıyor...")
            components = list(nx.connected_components(G))
            while len(components) > 1:
                G.add_edge(random.choice(list(components[0])), 
                          random.choice(list(components[1])))
                components = list(nx.connected_components(G))
        
        # Düğüm özellikleri ekle
        for node in G.nodes():
            G.nodes[node]['processing_delay'] = round(random.uniform(0.5, 2.0), 2)
            G.nodes[node]['reliability'] = round(random.uniform(0.95, 0.999), 4)
            G.nodes[node]['pos_x'] = random.uniform(0, 100)
            G.nodes[node]['pos_y'] = random.uniform(0, 100)
        
        # Bağlantı özellikleri ekle
        for u, v in G.edges():
            G.edges[u, v]['bandwidth'] = random.randint(100, 1000)
            G.edges[u, v]['delay'] = round(random.uniform(3, 15), 2)
            G.edges[u, v]['reliability'] = round(random.uniform(0.95, 0.999), 4)
        
        print(f"Ağ oluşturuldu: {len(G.nodes())} düğüm, {len(G.edges())} bağlantı")
        return G
    
    def save_network(self, G, filename="../data/mock_network.pkl"):  # DÜZELTİLDİ: ../data/
        """Ağı kaydet"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(G, f)
        print(f"Ağ kaydedildi: {filename}")
        
    def generate_sample_paths(self, G, source=0, target=49):
        """Örnek yollar oluştur (GA, ACO, RL için)"""
        print("Örnek yollar oluşturuluyor...")
        
        paths = {}
        
        # GA yolu (en kısa yol)
        try:
            ga_path = nx.shortest_path(G, source, target, weight='delay')
        except:
            ga_path = [source, random.choice(list(G.neighbors(source))), target]
            
        paths['GA'] = {
            'path': ga_path,
            'metrics': {
                'total_delay': round(random.uniform(30, 60), 2),
                'reliability_cost': round(random.uniform(0.01, 0.05), 4),
                'resource_cost': round(random.uniform(5, 15), 2)
            }
        }
        
        # ACO yolu
        middle_nodes = random.sample(list(G.nodes())[1:-1], min(4, self.n_nodes-2))
        paths['ACO'] = {
            'path': [source] + middle_nodes + [target],
            'metrics': {
                'total_delay': round(random.uniform(40, 70), 2),
                'reliability_cost': round(random.uniform(0.005, 0.04), 4),
                'resource_cost': round(random.uniform(3, 12), 2)
            }
        }
        
        # RL yolu
        rl_path = [source]
        current = source
        for _ in range(6):
            if current == target:
                break
            neighbors = list(G.neighbors(current))
            if neighbors:
                current = random.choice(neighbors)
                rl_path.append(current)
                
        if rl_path[-1] != target:
            rl_path.append(target)
            
        paths['RL'] = {
            'path': rl_path,
            'metrics': {
                'total_delay': round(random.uniform(35, 65), 2),
                'reliability_cost': round(random.uniform(0.01, 0.06), 4),
                'resource_cost': round(random.uniform(4, 14), 2)
            }
        }
        
        print(f"{len(paths)} algoritma için yollar oluşturuldu")
        return paths

# Ana program
if __name__ == "__main__":
    # 1. Generator'ü oluştur
    generator = MockNetworkGenerator(n_nodes=50)  # Test için 50 düğüm
    
    # 2. Ağı oluştur
    G = generator.generate()
    
    # 3. Kaydet - DÜZELTİLDİ: ../data/
    generator.save_network(G, "../data/mock_network.pkl")
    
    # 4. Örnek yollar oluştur
    paths = generator.generate_sample_paths(G)
    
    # 5. Yolları JSON olarak kaydet - DÜZELTİLDİ: ../data/
    os.makedirs("../data", exist_ok=True)
    with open("../data/sample_paths.json", "w", encoding='utf-8') as f:
        json.dump(paths, f, indent=2, ensure_ascii=False)
    
    print("Yollar kaydedildi: ../data/sample_paths.json")
    print("\nMock veriler başarıyla oluşturuldu!")
    print("\nİşlem tamam! Şimdi visualizer.py'yi çalıştırabilirsin.")