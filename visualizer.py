import matplotlib.pyplot as plt
import networkx as nx
import pickle
import json
import os
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

print("BSM307 - ENHANCED PATH VISUALIZATION ENGINEER")
print("=" * 70)
print("Mock veriler yüklendi!")
print("NetworkX çalışıyor!")
print("=" * 70)

class PathVisualizer:
    def __init__(self):
        self.G = None
        self.paths = None
        self.colors = {
            'GA': '#FF0000',
            'ACO': '#0000FF',
            'RL': '#00FF00'
        }
    
    def load_data(self):
        try:
            with open('../data/mock_network.pkl', 'rb') as f:
                self.G = pickle.load(f)
            print(f"Ağ yüklendi: {len(self.G.nodes())} düğüm")
            
            with open('../data/sample_paths.json', 'r') as f:
                self.paths = json.load(f)
            print(f"Yollar yüklendi: {list(self.paths.keys())}")
            return True
            
        except Exception as e:
            print(f"Hata: {e}")
            return False
    
    def get_node_positions(self):
        pos = {}
        for node in self.G.nodes():
            if 'pos_x' in self.G.nodes[node] and 'pos_y' in self.G.nodes[node]:
                pos[node] = (self.G.nodes[node]['pos_x'], 
                           self.G.nodes[node]['pos_y'])
            else:
                pos[node] = (node % 10 * 10, node // 10 * 10)
        return pos
    
    def visualize_single_path(self, algorithm='GA'):
        if algorithm not in self.paths:
            print(f"{algorithm} bulunamadı!")
            return
        
        print(f"\n{algorithm} yolu görselleştiriliyor...")
        plt.figure(figsize=(14, 12))
        pos = self.get_node_positions()
        path = self.paths[algorithm]['path']
        path_edges = list(zip(path[:-1], path[1:]))
        
        nx.draw_networkx_nodes(self.G, pos, node_size=50,
                              node_color='lightgray', alpha=0.6)
        nx.draw_networkx_edges(self.G, pos, edge_color='#CCCCCC',
                              width=0.8, alpha=0.4)
        
        nx.draw_networkx_edges(self.G, pos, edgelist=path_edges,
                              edge_color=self.colors[algorithm],
                              width=4, alpha=0.9)
        
        nx.draw_networkx_nodes(self.G, pos, nodelist=path,
                              node_size=200, node_color=self.colors[algorithm],
                              edgecolors='black', linewidths=1.5, alpha=0.9)
        
        nx.draw_networkx_nodes(self.G, pos, nodelist=[path[0], path[-1]],
                              node_size=250, node_color=['lime', 'red'],
                              edgecolors='black', linewidths=2)
        
        plt.title(f'{algorithm} Algorithm Path', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        os.makedirs('outputs', exist_ok=True)
        filename = f'outputs/path_{algorithm.lower()}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Görsel kaydedildi: {filename}")
        plt.show()
    
    def visualize_comparison(self):
        print("\nAlgoritma karşılaştırması...")
        plt.figure(figsize=(16, 14))
        pos = self.get_node_positions()
        
        nx.draw_networkx_nodes(self.G, pos, node_size=30,
                              node_color='lightgray', alpha=0.5)
        nx.draw_networkx_edges(self.G, pos, edge_color='#DDDDDD',
                              width=0.5, alpha=0.3)
        
        line_styles = {'GA': 'solid', 'ACO': 'dashed', 'RL': 'dotted'}
        
        for algo in ['GA', 'ACO', 'RL']:
            if algo in self.paths:
                path = self.paths[algo]['path']
                path_edges = list(zip(path[:-1], path[1:]))
                
                nx.draw_networkx_edges(self.G, pos, edgelist=path_edges,
                                      edge_color=self.colors[algo], width=3,
                                      style=line_styles[algo], alpha=0.7,
                                      label=f'{algo} Path')
        
        plt.legend(fontsize=12, loc='upper right')
        plt.title('Multi-Algorithm Path Comparison\nGA (Red) | ACO (Blue) | RL (Green)',
                 fontsize=18, fontweight='bold')
        plt.axis('off')
        
        filename = 'outputs/comparison.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma görseli kaydedildi: {filename}")
        plt.show()
    
    def show_path_metrics(self):
        print("\nYol Metrikleri:")
        print("-" * 50)
        
        for algo, data in self.paths.items():
            metrics = data['metrics']
            print(f"{algo}:")
            print(f"  Yol: {data['path']}")
            print(f"  Uzunluk: {len(data['path'])} düğüm")
            print(f"  Toplam Gecikme: {metrics['total_delay']} ms")
            print(f"  Güvenilirlik Maliyeti: {metrics['reliability_cost']}")
            print(f"  Kaynak Maliyeti: {metrics['resource_cost']}")
            print()
    
    # GÖREV 4: Tooltip sistemi
    def create_path_tooltip_visualization(self):
        """Algoritma yolları için tooltip'li görselleştirme"""
        print("\nAlgoritma yolları için detaylı görselleştirme...")
        
        for algo in ['GA', 'ACO', 'RL']:
            if algo not in self.paths:
                continue
                
            plt.figure(figsize=(14, 10))
            pos = self.get_node_positions()
            path = self.paths[algo]['path']
            
            nx.draw_networkx_nodes(self.G, pos, node_size=30, 
                                  node_color='lightgray', alpha=0.3)
            nx.draw_networkx_edges(self.G, pos, edge_color='#DDDDDD', 
                                  width=0.5, alpha=0.2)
            
            for idx, node in enumerate(path):
                if idx == 0:
                    node_color = 'lime'
                    node_size = 250
                elif idx == len(path) - 1:
                    node_color = 'red'
                    node_size = 250
                else:
                    node_color = self.colors[algo]
                    node_size = 200
                
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node],
                                      node_size=node_size,
                                      node_color=node_color,
                                      edgecolors='black', linewidths=1.5)
                
                delay = self.G.nodes[node].get('processing_delay', 'N/A')
                plt.text(pos[node][0], pos[node][1] + 4,
                        f"Node {node}\nDelay: {delay}ms", fontsize=8,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor="lightblue", alpha=0.7))
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.G.has_edge(u, v):
                    nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)],
                                          edge_color=self.colors[algo],
                                          width=3, alpha=0.8)
                    try:    
                        edge_info = self.G.edges[u, v]
                        bw = edge_info.get('bandwidth', 'N/A')
                        link_delay = edge_info.get('delay', 'N/A')
                    
                        mid_x = (pos[u][0] + pos[v][0]) / 2
                        mid_y = (pos[u][1] + pos[v][1]) / 2
                    
                        plt.text(mid_x, mid_y, f"BW: {bw}\nDelay: {link_delay}ms",
                            fontsize=7, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor="yellow", alpha=0.7))
                    except:
                        pass  # Edge varsa ama veri yoksa atla
            
            plt.title(f'{algo} Path with Metrics', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            filename = f'outputs/path_{algo.lower()}_with_metrics.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ {algo} metrik görseli: {filename}")
    
    def create_interactive_network(self):
        """GÖREV 4: İnteraktif ağ haritası"""
        print("\n" + "="*60)
        print("GÖREV 4: İnteraktif Ağ Haritası Oluşturuluyor")
        print("="*60)
        
        try:
            pos = self.get_node_positions()
            
            node_x, node_y = [], []
            node_text = []
            
            for node in self.G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                info = f"""
                <b>DÜĞÜM {node}</b><br>
                <b>Processing Delay:</b> {self.G.nodes[node].get('processing_delay', 'N/A')}ms<br>
                <b>Reliability:</b> {self.G.nodes[node].get('reliability', 'N/A')}<br>
                <b>Position:</b> ({x:.1f}, {y:.1f})
                """
                node_text.append(info)
            
            edge_x, edge_y = [], []
            
            for u, v in self.G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(width=0.8, color='#888888'), hoverinfo='none'
            ))
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode='markers',
                marker=dict(size=10, color='lightblue'),
                text=node_text, hoverinfo='text'
            ))
            
            fig.update_layout(
                title='Interactive Network Map (Hover for details)',
                showlegend=False, hovermode='closest',
                plot_bgcolor='white', width=1000, height=800
            )
            
            os.makedirs('outputs', exist_ok=True)
            fig.write_html("outputs/interactive_network.html")
            print("✓ İnteraktif harita: outputs/interactive_network.html")
            
        except Exception as e:
            print(f"Interaktif harita hatası: {e}")
    
    def create_metrics_comparison_chart(self):
        """Metrik karşılaştırma grafiği"""
        print("\nMetrik karşılaştırma grafikleri oluşturuluyor...")
        
        algorithms = []
        delays = []
        reliability_costs = []
        resource_costs = []
        path_lengths = []
        
        for algo in ['GA', 'ACO', 'RL']:
            if algo in self.paths:
                algorithms.append(algo)
                metrics = self.paths[algo]['metrics']
                delays.append(metrics['total_delay'])
                reliability_costs.append(metrics['reliability_cost'])
                resource_costs.append(metrics['resource_cost'])
                path_lengths.append(len(self.paths[algo]['path']))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = [self.colors[a] for a in algorithms]
        
        axes[0,0].bar(algorithms, delays, color=colors, edgecolor='black')
        axes[0,0].set_title('Total Delay Comparison', fontweight='bold')
        axes[0,0].set_ylabel('Delay (ms)')
        
        axes[0,1].bar(algorithms, reliability_costs, color=colors, edgecolor='black')
        axes[0,1].set_title('Reliability Cost Comparison', fontweight='bold')
        
        axes[1,0].bar(algorithms, resource_costs, color=colors, edgecolor='black')
        axes[1,0].set_title('Resource Cost Comparison', fontweight='bold')
        axes[1,0].set_ylabel('Cost')
        
        axes[1,1].bar(algorithms, path_lengths, color=colors, edgecolor='black')
        axes[1,1].set_title('Path Length Comparison', fontweight='bold')
        axes[1,1].set_ylabel('Number of Nodes')
        
        plt.suptitle('Algorithm Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'outputs/metrics_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Metrik karşılaştırma: {filename}")
    
    def generate_pdf_report(self):
        """GÖREV 5: PDF raporu oluştur"""
        print("\n" + "="*60)
        print("GÖREV 5: PDF Raporu Oluşturuluyor")
        print("="*60)
        
        try:
            pdf = canvas.Canvas("outputs/path_analysis_report.pdf", pagesize=letter)
            width, height = letter
            
            pdf.setFont("Helvetica-Bold", 24)
            pdf.drawString(100, height-150, "BSM307 - Path Analysis Report")
            pdf.setFont("Helvetica", 14)
            pdf.drawString(100, height-180, "Meta-Heuristic & Reinforcement Learning Project")
            
            pdf.setFont("Helvetica", 12)
            pdf.drawString(100, height-220, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            pdf.drawString(100, height-240, f"Network Nodes: {len(self.G.nodes())}")
            pdf.drawString(100, height-260, f"Network Edges: {len(self.G.edges())}")
            
            pdf.showPage()
            
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(100, height-100, "Algorithm Performance Comparison")
            
            y_pos = height-150
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(100, y_pos, "Algorithm  Delay (ms)  Reliability  Cost  Length")
            pdf.line(100, y_pos-5, 500, y_pos-5)
            
            y_pos -= 25
            pdf.setFont("Helvetica", 11)
            
            for algo in ['GA', 'ACO', 'RL']:
                if algo in self.paths:
                    metrics = self.paths[algo]['metrics']
                    path_len = len(self.paths[algo]['path'])
                    
                    row = f"{algo:10} {metrics['total_delay']:10.2f} {metrics['reliability_cost']:12.4f} {metrics['resource_cost']:7.2f} {path_len:7}"
                    pdf.drawString(100, y_pos, row)
                    y_pos -= 20
            
            pdf.showPage()
            
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(100, height-100, "Visualization Outputs")
            
            pdf.setFont("Helvetica", 12)
            pdf.drawString(100, height-150, "Generated Files:")
            pdf.drawString(100, height-170, "1. outputs/path_ga.png - GA Algorithm Path")
            pdf.drawString(100, height-190, "2. outputs/path_aco.png - ACO Algorithm Path")
            pdf.drawString(100, height-210, "3. outputs/path_rl.png - RL Algorithm Path")
            pdf.drawString(100, height-230, "4. outputs/comparison.png - Algorithm Comparison")
            pdf.drawString(100, height-250, "5. outputs/interactive_network.html - Interactive Map")
            
            pdf.save()
            print("✓ PDF raporu: outputs/path_analysis_report.pdf")
            
        except Exception as e:
            print(f"PDF oluşturma hatası: {e}")

def main():
    print("\n" + "="*70)
    print("BSM307 - ENHANCED PATH VISUALIZATION SYSTEM")
    print("="*70)
    
    viz = PathVisualizer()
    
    if not viz.load_data():
        print("Veriler yüklenemedi!")
        return
    
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*50)
    print("ORİJİNAL GÖREVLER (1-3)")
    print("="*50)
    
    for algo in ['GA', 'ACO', 'RL']:
        viz.visualize_single_path(algo)
    
    viz.visualize_comparison()
    
    print("\n" + "="*50)
    print("YENİ GÖREVLER (4-5)")
    print("="*50)
    
    viz.create_path_tooltip_visualization()
    viz.create_interactive_network()
    viz.create_metrics_comparison_chart()
    viz.generate_pdf_report()
    
    viz.show_path_metrics()
    
    print("\n" + "="*70)
    print("TÜM GÖREVLER BAŞARIYLA TAMAMLANDI! ")
    print("="*70)
    print("\nOLUŞAN DOSYALAR:")
    print("  outputs/path_ga.png                   - GA yolu")
    print("  outputs/path_aco.png                  - ACO yolu") 
    print("  outputs/path_rl.png                   - RL yolu")
    print("  outputs/comparison.png               - Karşılaştırma")
    print("  outputs/interactive_network.html     - İnteraktif harita (GÖREV 4)")
    print("  outputs/path_*_with_metrics.png      - Metrikli yollar")
    print("  outputs/metrics_comparison.png       - Metrik grafikleri")
    print("  outputs/path_analysis_report.pdf     - PDF rapor (GÖREV 5)")
    print("\nProjenin görselleştirme kısmı TAMAMDIR!")

if __name__ == "__main__":
    main()