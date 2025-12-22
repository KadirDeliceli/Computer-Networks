import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

# --- DUMMY / IMPORT BLOKLARI ---
# --- IMPORT BLOKLARI ---
try:
    # Kullanıcının modülünden fonksiyonları çekmeye çalış
    from mehdituncer.ag_ve_metrikler import create_random_graph, compute_metrics
    print("Başarılı: 'mt ap topolojisinden modülünden fonksiyonlar yüklendi.")
except ImportError as e:
    print(f"Uyarı: Kullanıcı modülü yüklenemedi ({e}). Dummy veriler kullanılacak.")
    import random

    def create_random_graph(num_nodes, p, **kwargs):
        print("KULLANILIYOR: Dummy create_random_graph")
        return nx.erdos_renyi_graph(n=num_nodes, p=p, seed=42)

    def compute_metrics(G, path):
        print("KULLANILIYOR: Dummy compute_metrics")
        return {'total_delay': 5.5, 'total_reliability': 0.99, 'resource_cost': 12.0}

# GA Algoritması (Henüz implemente edilmemiş olabilir, dummy kalacak)
try:
    from ga import run_genetic_algorithm
except ImportError:
    def run_genetic_algorithm(G, s, d, demand, weights):
        try:
            # Sadece test için en kısa yolu bul
            path = nx.shortest_path(G, s, d)
            return path, 10.5
        except:
            return None, 0


class NetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ağ Topolojisi")
        self.root.geometry("1200x900")

        # Grafik verileri
        self.G = None
        self.pos = None
        self.current_path = None

        # --- GÖRSEL AYAR DEĞİŞKENLERİ ---
        self.var_node_size = tk.DoubleVar(value=50.0)
        self.var_edge_width = tk.DoubleVar(value=1.0)
        self.var_edge_alpha = tk.DoubleVar(value=0.6)

        # --- ARAYÜZ DÜZENİ ---

        # Sol Panel (Kontroller) - Rengine dokunulmuyor
        control_frame = ttk.Frame(root, padding="15")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 1. Grafik Oluşturma
        ttk.Label(control_frame, text="Ağ Topolojisi", font=("Arial", 12, "bold")).pack(pady=5)
        frm_gen = ttk.Frame(control_frame);
        frm_gen.pack(fill='x')
        ttk.Label(frm_gen, text="Düğüm Sayısı:").grid(row=0, column=0, sticky='w')
        self.entry_nodes = ttk.Entry(frm_gen, width=10);
        self.entry_nodes.insert(0, "50");
        self.entry_nodes.grid(row=0, column=1, padx=5)
        ttk.Label(frm_gen, text="Bağlantı (P):").grid(row=1, column=0, sticky='w')
        self.entry_prob = ttk.Entry(frm_gen, width=10);
        self.entry_prob.insert(0, "0.15");
        self.entry_prob.grid(row=1, column=1, padx=5)
        self.btn_create = ttk.Button(control_frame, text="YENİ AĞ OLUŞTUR", command=self.generate_graph);
        self.btn_create.pack(pady=15, fill='x')

        # --- GÖRSEL AYARLAR PANELİ ---
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(control_frame, text="Görsel Ayarlar", font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Label(control_frame, text="Düğüm Boyutu:").pack(anchor='w')
        self.scale_node_size = ttk.Scale(control_frame, from_=10, to=300, variable=self.var_node_size,
                                         command=self.on_visual_change);
        self.scale_node_size.pack(fill='x', pady=(0, 10))
        ttk.Label(control_frame, text="Çizgi Kalınlığı:").pack(anchor='w')
        self.scale_edge_width = ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.var_edge_width,
                                          command=self.on_visual_change);
        self.scale_edge_width.pack(fill='x', pady=(0, 10))
        ttk.Label(control_frame, text="Çizgi Görünürlüğü (Saydamlık):").pack(anchor='w')
        self.scale_edge_alpha = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.var_edge_alpha,
                                          command=self.on_visual_change);
        self.scale_edge_alpha.pack(fill='x', pady=(0, 10))
        self.var_show_edges = tk.BooleanVar(value=True)
        self.chk_show_edges = ttk.Checkbutton(control_frame, text="Tüm Bağlantıları Göster",
                                              variable=self.var_show_edges, command=self.on_visual_change_no_arg);
        self.chk_show_edges.pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 2. Rota Bulma
        ttk.Label(control_frame, text="QoS Rotalama", font=("Arial", 12, "bold")).pack(pady=5)
        frm_route = ttk.Frame(control_frame);
        frm_route.pack(fill='x')
        ttk.Label(frm_route, text="Kaynak (Start):").grid(row=0, column=0, sticky='w')
        self.entry_s = ttk.Entry(frm_route, width=8);
        self.entry_s.insert(0, "0");
        self.entry_s.grid(row=0, column=1)
        ttk.Label(frm_route, text="Hedef (End):").grid(row=1, column=0, sticky='w')
        self.entry_d = ttk.Entry(frm_route, width=8);
        self.entry_d.insert(0, "10");
        self.entry_d.grid(row=1, column=1)
        ttk.Label(frm_route, text="Bant Genişliği:").grid(row=2, column=0, sticky='w')
        self.entry_demand = ttk.Entry(frm_route, width=8);
        self.entry_demand.insert(0, "150");
        self.entry_demand.grid(row=2, column=1)
        ttk.Label(control_frame, text="Optimizasyon Kriterleri:", font=("Arial", 9, "italic")).pack(pady=(10, 5))
        frm_weights = ttk.Frame(control_frame);
        frm_weights.pack()
        ttk.Label(frm_weights, text="Gecikme").grid(row=0, column=0);
        self.entry_w_delay = ttk.Entry(frm_weights, width=5);
        self.entry_w_delay.insert(0, "0.4");
        self.entry_w_delay.grid(row=1, column=0, padx=2)
        ttk.Label(frm_weights, text="Güven").grid(row=0, column=1);
        self.entry_w_rel = ttk.Entry(frm_weights, width=5);
        self.entry_w_rel.insert(0, "0.3");
        self.entry_w_rel.grid(row=1, column=1, padx=2)
        ttk.Label(frm_weights, text="Maliyet").grid(row=0, column=2);
        self.entry_w_res = ttk.Entry(frm_weights, width=5);
        self.entry_w_res.insert(0, "0.3");
        self.entry_w_res.grid(row=1, column=2, padx=2)
        self.btn_find = ttk.Button(control_frame, text="EN İYİ ROTA HESAPLA", command=self.find_path);
        self.btn_find.pack(pady=15, fill='x')

        # Sonuç Paneli
        self.lbl_result = ttk.Label(control_frame, text="Hazır.", justify=tk.LEFT, background="#e1e1e1", padding=5);
        self.lbl_result.pack(fill='x')

        # Sağ Panel (Matplotlib Grafiği)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def generate_graph(self):
        try:
            n = int(self.entry_nodes.get())
            p = float(self.entry_prob.get())

            self.G = create_random_graph(num_nodes=n, p=p)
            self.pos = nx.spring_layout(self.G, seed=42, iterations=50)
            self.current_path = None

            self.draw_graph(highlight_path=None)
            self.lbl_result.config(text=f"{n} düğümlü ağ oluşturuldu.")
        except Exception as e:
            messagebox.showerror("Hata", f"Graf hatası: {str(e)}")

    def on_visual_change(self, val):
        self.draw_graph(highlight_path=self.current_path)

    def on_visual_change_no_arg(self):
        self.draw_graph(highlight_path=self.current_path)

    def draw_graph(self, highlight_path=None):
        self.ax.clear()

        # --- ARKA PLAN RENGİNİ AYARLA (Çok Açık Sarı) ---
        self.ax.set_facecolor('#FFFFE0')  # LightYellow

        if self.G is None:
            self.canvas.draw()
            return

        # Ayarları oku
        current_node_size = self.var_node_size.get()
        current_edge_width = self.var_edge_width.get()
        current_edge_alpha = self.var_edge_alpha.get()

        # 1. DÜĞÜM ÇİZİMİ (Mor - Değişmedi)
        nx.draw_networkx_nodes(
            self.G, self.pos,
            ax=self.ax,
            node_size=current_node_size,
            node_color='purple',
            edgecolors='black',
            linewidths=0.5
        )

        # 2. KENAR ÇİZİMİ (Kahverengi - Değişti)
        if self.var_show_edges.get():
            nx.draw_networkx_edges(
                self.G, self.pos,
                ax=self.ax,
                alpha=current_edge_alpha,
                width=current_edge_width,
                edge_color='brown'  # Çizgiler artık kahverengi
            )

        # 3. ROTA ÇİZİMİ (Vurgu - Değişmedi)
        if highlight_path:
            path_edges = list(zip(highlight_path, highlight_path[1:]))

            # Yol Düğümleri
            nx.draw_networkx_nodes(
                self.G, self.pos,
                nodelist=highlight_path,
                node_size=current_node_size * 1.3,
                node_color='purple',
                edgecolors='red',
                linewidths=2.0,
                ax=self.ax
            )

            # Yol Kenarları (Kırmızı vurgu)
            nx.draw_networkx_edges(
                self.G, self.pos,
                edgelist=path_edges,
                edge_color='red',
                width=3.0,
                alpha=1.0,
                ax=self.ax
            )

            # Start / End
            s, d = highlight_path[0], highlight_path[-1]
            se_size = current_node_size * 2.0
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[s], node_size=se_size, node_color='green',
                                   edgecolors='black', label='Start', ax=self.ax)
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[d], node_size=se_size, node_color='orange',
                                   edgecolors='black', label='End', ax=self.ax)

            self.ax.legend(loc='upper right', facecolor='white')

        self.ax.set_title("Ağ Topolojisi", fontsize=14, fontweight='bold')
        self.ax.axis('off')
        self.canvas.draw()

    def find_path(self):
        if self.G is None:
            messagebox.showwarning("Uyarı", "Önce ağı oluşturun!")
            return

        try:
            s = int(self.entry_s.get())
            d = int(self.entry_d.get())
            talep = float(self.entry_demand.get())
            w_delay = float(self.entry_w_delay.get());
            w_rel = float(self.entry_w_rel.get());
            w_res = float(self.entry_w_res.get())
            weights = {"delay": w_delay, "reliability": w_rel, "resource": w_res}

            best_path, best_cost = run_genetic_algorithm(self.G, s, d, talep, weights)

            if best_path:
                self.current_path = best_path
                metrics = compute_metrics(self.G, best_path)
                result_text = (f"ROTA BULUNDU!\nYol: {best_path}\nFitness: {best_cost:.4f}\n"
                               f"Gecikme: {metrics['total_delay']} ms\nGüvenilirlik: %{metrics['total_reliability'] * 100:.1f}")
                self.lbl_result.config(text=result_text, foreground="green")
                self.draw_graph(highlight_path=best_path)
            else:
                self.current_path = None
                self.lbl_result.config(text="Uygun yol bulunamadı!", foreground="red")
                self.draw_graph(highlight_path=None)

        except Exception as e:
            messagebox.showerror("Hata", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkApp(root)
    root.mainloop()