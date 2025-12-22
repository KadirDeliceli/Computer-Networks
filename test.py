import sys
sys.path.append('src')

try:
    from visualizer import PathVisualizer
    print("PathVisualizer import başarılı!")
    
    viz = PathVisualizer()
    print("PathVisualizer oluşturuldu!")
    
    # Basit test
    if viz.load_data():
        print("Veriler yüklendi!")
        print(f"  - Nodes: {len(viz.G.nodes())}")
        print(f"  - Paths: {list(viz.paths.keys())}")
    else:
        print("Veri yükleme hatası!")
        
except Exception as e:
    print(f"Hata: {e}")