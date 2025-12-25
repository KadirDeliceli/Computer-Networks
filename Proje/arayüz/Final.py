import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

from genetik.ga import run_genetic_algorithm
from graph_and_metriks.graph_utils import create_random_graph
from graph_and_metriks.metrics import compute_metrics

from qlearning.Qlearning import Q_Learning_run


class QoSRoutingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QoS Tabanlı Çok Amaçlı Rotalama")
        self.root.geometry("1500x900")
        self.root.configure(bg="#f8f9fa")

        # ===== main2: Grafik oluşturma =====
        self.G = create_random_graph(
            250, 0.03,
            edge_file="C:\\Users\\dkadi\\OneDrive - Bartın Üniversitesi\\Masaüstü\\Bilgisayar Ağları\\BSM307_317_Guz2025_TermProject_EdgeData.csv",
            demand_file="C:\\Users\\dkadi\\OneDrive - Bartın Üniversitesi\\Masaüstü\\Bilgisayar Ağları\\BSM307_317_Guz2025_TermProject_DemandData.csv",
            node_file="C:\\Users\\dkadi\\OneDrive - Bartın Üniversitesi\\Masaüstü\\Bilgisayar Ağları\\BSM307_317_Guz2025_TermProject_NodeData.csv"
        )

        self.pos = nx.spring_layout(self.G, seed=42)
        self.current_path = None

        # ===== main4: Görsel ayar değişkenleri =====
        self.var_node_size = tk.DoubleVar(value=40)
        self.var_edge_width = tk.DoubleVar(value=0.1)
        self.var_edge_alpha = tk.DoubleVar(value=1.0)
        self.var_show_edges = tk.BooleanVar(value=True)

        # ===== Default değerler =====
        self.src = tk.IntVar(value=8)
        self.dst = tk.IntVar(value=44)
        self.dem = tk.DoubleVar(value=950)

        # ===== Algoritma seçimi =====
        self.algorithm_var = tk.StringVar(value="Genetic Algorithm")

        self.build_ui()
        self.draw_graph()

    # ================= UI =================
    def build_ui(self):
        # -------- SOL PANEL (main4) --------
        left = tk.Frame(self.root, width=340, bg="white", padx=12)
        left.pack_propagate(False)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Kullanıcı Girişleri",
                 font=("Segoe UI", 16, "bold"),
                 bg="white").pack(pady=8)

        self._label(left, "Kaynak Düğüm (S)")
        tk.Spinbox(left, from_=0, to=249, textvariable=self.src).pack(fill=tk.X)

        self._label(left, "Hedef Düğüm (D)")
        tk.Spinbox(left, from_=0, to=249, textvariable=self.dst).pack(fill=tk.X)

        self._label(left, "Talep (Bandwidth)")
        tk.Entry(left, textvariable=self.dem).pack(fill=tk.X)

        # ===== main2: Optimizasyon ağırlıkları =====
        tk.Label(left, text="\nOptimizasyon Ağırlıkları",
                 font=("Segoe UI", 11, "bold"),
                 bg="white").pack(anchor="w")

        self.w1 = self._weight_slider(left, "Gecikme", 0.4)
        self.w2 = self._weight_slider(left, "Güvenilirlik", 0.3)
        self.w3 = self._weight_slider(left, "Kaynak", 0.3)

        ttk.Separator(left).pack(fill="x", pady=8)

        # ===== main4: Görsel ayarlar =====
        tk.Label(left, text="Görsel Ayarlar",
                 font=("Segoe UI", 11, "bold"),
                 bg="white").pack(anchor="w")

        self._visual_slider(left, "Node Boyutu", self.var_node_size, 10, 300)
        self._visual_slider(left, "Kenar Kalınlığı", self.var_edge_width, 0.1, 5)
        self._visual_slider(left, "Kenar Saydamlığı", self.var_edge_alpha, 0, 1)

        ttk.Checkbutton(left, text="Tüm Kenarları Göster",
                        variable=self.var_show_edges,
                        command=self.draw_graph).pack(anchor="w", pady=3)

        ttk.Separator(left).pack(fill="x", pady=8)

        # ===== Algoritma seçimi =====
        tk.Label(left, text="Algoritma Seçimi",
                 font=("Segoe UI", 11, "bold"),
                 bg="white").pack(anchor="w")

        ttk.Combobox(
            left,
            textvariable=self.algorithm_var,
            values=["Genetic Algorithm", "Q-Learning"],
            state="readonly"
        ).pack(fill=tk.X, pady=4)

        tk.Button(left, text="ALGORİTMAYI ÇALIŞTIR",
                  bg="#20c997", fg="white",
                  font=("Segoe UI", 11, "bold"),
                  command=self.run_selected_algorithm).pack(fill=tk.X, pady=6)

        # ===== Sonuç metni =====
        tk.Label(left, text="Sonuç Detayı",
                 font=("Segoe UI", 11, "bold"),
                 bg="white").pack(anchor="w")

        self.result_text = tk.Text(left, height=8, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X)

        # -------- SAĞ PANEL --------
        right = tk.Frame(self.root, bg="#f8f9fa")
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # ===== main4: Kartlar (SAĞDA – AŞAĞIDAN YUKARI) =====
        # ===== main4: Kartlar (sağda, dikey, daha büyük) =====
        cards = tk.Frame(right, bg="#f8f9fa", width=300)
        cards.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
        cards.pack_propagate(False)

        self.card_res = self._create_card(cards, "Kaynak", "0", "#339af0")
        self.card_rel = self._create_card(cards, "Güvenilirlik", "%0", "#51cf66")
        self.card_delay = self._create_card(cards, "Ort. Gecikme", "0 ms", "#ff6b6b")

        # ===== Grafik alanı =====
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    # ================= HELPERS =================
    def _label(self, p, t):
        tk.Label(p, text=t, bg="white").pack(anchor="w", pady=(4, 0))

    def _weight_slider(self, p, t, d):
        tk.Label(p, text=t, bg="white").pack(anchor="w")
        v = tk.DoubleVar(value=d)
        tk.Scale(p, from_=0, to=1, resolution=0.1,
                 orient=tk.HORIZONTAL,
                 variable=v, bg="white").pack(fill=tk.X)
        return v

    def _visual_slider(self, p, t, var, f, to):
        tk.Label(p, text=t, bg="white").pack(anchor="w")
        ttk.Scale(p, from_=f, to=to,
                  variable=var,
                  command=lambda e: self.draw_graph()).pack(fill=tk.X)

    def _create_card(self, p, title, val, color):
        c = tk.Frame(p, bg="white", padx=30, pady=24)
        c.pack(fill=tk.X, pady=12)

        tk.Label(
            c,
            text=title,
            bg="white",
            fg="#868e96",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=(0, 6))

        lbl = tk.Label(
            c,
            text=val,
            bg="white",
            fg=color,
            font=("Segoe UI", 22, "bold")
        )
        lbl.pack()

        return lbl

    # ===== main2: Ağırlık normalizasyonu =====
    def get_weights(self):
        s = self.w1.get() + self.w2.get() + self.w3.get()
        if s == 0:
            return None
        return {
            "delay": self.w1.get() / s,
            "reliability": self.w2.get() / s,
            "resource": self.w3.get() / s
        }

    # ================= ALGORITHM =================
    def run_selected_algorithm(self):
        s = self.src.get()
        d = self.dst.get()
        if s not in self.G.nodes or d not in self.G.nodes:
            messagebox.showerror(
                "Geçersiz Düğüm",
                "Girilen kaynak (S) veya hedef (D) düğüm graf içerisinde bulunmamaktadır."
            )
            self.current_path = None
            self.SonucYazdir(0, 0,0 , self.algorithm_var.get() , 0, float('inf'), [])
            return

        algo = self.algorithm_var.get()
        weights = self.get_weights()

        if not weights:
            messagebox.showerror("Hata", "Ağırlıklar geçersiz")
            return

        start = time.perf_counter()

        if algo == "Genetic Algorithm":
            path, cost = run_genetic_algorithm(
                self.G, self.src.get(), self.dst.get(),
                self.dem.get(), weights
            )
        else:  # Q-Learning
            path, cost = Q_Learning_run(
                self.G, self.src.get(), self.dst.get(),
                self.dem.get(), weights
            )

        elapsed = time.perf_counter() - start

        self.current_path = path
        m = compute_metrics(self.G, path)

        if(m is None):
            td = 0
            tr = 0
            rc = 0
        else:
            td=m["total_delay"]
            tr=m["total_reliability"]
            rc=m["resource_cost"]

        if path is None or len(path) < 2:
            messagebox.showwarning(
                "Yol Üretilemedi",
                "Seçilen kaynak ve hedef arasında geçerli bir yol üretilemedi."
            )
            self.current_path = None

        self.SonucYazdir(td, tr, rc, algo, elapsed, cost, path)



    def SonucYazdir(self,td,tr,rc,algo,elapsed,cost,path):
        self.card_delay.config(text=f"{td:.2f} ms")
        self.card_rel.config(text=f"%{tr * 100:.1f}")
        self.card_res.config(text=f"{rc:.1f}")

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Algoritma: {algo}\n")
        self.result_text.insert(tk.END, f"Süre: {elapsed:.4f} sn\n")
        self.result_text.insert(tk.END, f"Fitness: {cost:.4f}\n")
        self.result_text.insert(tk.END, f"Yol: {path}\n")

        self.ax.set_title(f"{algo} | Süre: {elapsed:.3f} sn")
        self.draw_graph()

    # ================= DRAW =================
    def draw_graph(self):
        self.ax.clear()
        self.ax.set_facecolor("#FFFFE0")

        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax,
            node_size=self.var_node_size.get(),
            node_color="purple",
            edgecolors="black"
        )

        if self.var_show_edges.get():
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                width=self.var_edge_width.get(),
                alpha=self.var_edge_alpha.get(),
                edge_color="brown"
            )

        if self.current_path:
            edges = list(zip(self.current_path, self.current_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=edges,
                edge_color="red", width=3, ax=self.ax
            )

        self.ax.axis("off")
        self.canvas.draw()


# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    QoSRoutingApp(root)
    root.mainloop()
