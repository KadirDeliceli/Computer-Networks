import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import pandas as pd

from genetik_ga import run_genetic_algorithm
from graph_utils import create_random_graph
from metrics import compute_metrics

from Qlearning import Q_Learning_run


class QoSRoutingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QoS Tabanlı Çok Amaçlı Rotalama (Modern UI)")
        
        # ==========================================
        # 1. APPLE DESIGN SYSTEM CONFIGURATION
        # ==========================================
        self.font_scale = 1.2  # Change this to 1.0, 1.2, 1.5 to scale the whole UI
        
        self.colors = {
            "bg_main": "#F2F2F7",       # System Gray 6 (App Background)
            "bg_card": "#FFFFFF",       # Pure White (Card Background)
            "accent": "#007AFF",        # System Blue (Primary Action)
            "accent_hover": "#005BB5",  # Darker Blue
            "text_main": "#1C1C1E",     # Label Color
            "text_secondary": "#8E8E93",# Secondary Label
            "success": "#34C759",       # System Green
            "error": "#FF3B30",         # System Red
            "graph_node": "#E5E5EA",    # System Gray 5
            "graph_edge": "#C7C7CC"     # System Gray 4
        }

        # Set Window Size and Background
        self.root.geometry("1600x1000")
        self.root.configure(bg=self.colors["bg_main"])

        # Initialize Style Engine
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam' allows easier custom coloring than 'vista'
        self.configure_styles()

        # ==========================================
        # 2. DATA LOADING (Original Paths Kept)
        # ==========================================
        # Note: Ensure these paths are correct on your machine
        self.G = create_random_graph(
            250, 0.03,
            edge_file="../data/BSM307_317_Guz2025_TermProject_EdgeData.csv",
            demand_file="../data/BSM307_317_Guz2025_TermProject_DemandData.csv",
            node_file="../data/BSM307_317_Guz2025_TermProject_NodeData.csv"
        )

        self.pos = nx.spring_layout(self.G, seed=42)
        self.current_path = None

        # ==========================================
        # 3. VARIABLES
        # ==========================================
        self.var_node_size = tk.DoubleVar(value=60)
        self.var_edge_width = tk.DoubleVar(value=0.5)
        self.var_edge_alpha = tk.DoubleVar(value=0.6)
        self.var_show_edges = tk.BooleanVar(value=True)

        self.src = tk.IntVar(value=8)
        self.dst = tk.IntVar(value=44)
        self.dem = tk.DoubleVar(value=950)

        self.algorithm_var = tk.StringVar(value="Genetic Algorithm")

        # ==========================================
        # 4. BUILD UI & DRAW
        # ==========================================
        self.build_ui()
        self.draw_graph()

    def configure_styles(self):
        """Defines modern ttk styles using the color palette."""
        base_size = int(11 * self.font_scale)
        
        # Main Button Style
        self.style.configure(
            "Apple.TButton",
            font=("Segoe UI", base_size, "bold"),
            background=self.colors["accent"],
            foreground="white",
            borderwidth=0,
            focuscolor=self.colors["accent"],
            padding=(int(10 * self.font_scale), int(10 * self.font_scale))
        )
        self.style.map("Apple.TButton", 
                       background=[('active', self.colors["accent_hover"])])

        # Combobox Style
        self.style.configure(
            "TCombobox", 
            fieldbackground="white", 
            background="white",
            selectbackground=self.colors["bg_main"],
            arrowcolor=self.colors["text_main"],
            font=("Segoe UI", base_size)
        )

        # Horizontal Scale (Slider)
        self.style.configure(
            "Horizontal.TScale",
            background=self.colors["bg_card"],
            troughcolor=self.colors["bg_main"],
            sliderlength=int(20 * self.font_scale)
        )
        
        # Checkbutton
        self.style.configure(
            "TCheckbutton",
            background=self.colors["bg_card"],
            font=("Segoe UI", base_size),
            foreground=self.colors["text_main"]
        )

    def build_ui(self):
        """Constructs the two-panel layout."""
        
        # ---------------- LEFT PANEL (CONTROLS) ----------------
        # Dynamic width based on font_scale
        panel_width = int(380 * self.font_scale)
        
        left = tk.Frame(self.root, width=panel_width, bg=self.colors["bg_card"], 
                        padx=int(25 * self.font_scale), pady=int(25 * self.font_scale))
        left.pack_propagate(False) # Force fixed width
        left.pack(side=tk.LEFT, fill=tk.Y)

        # App Title
        tk.Label(left, text="QoS Routing", 
                 font=("Segoe UI", int(22 * self.font_scale), "bold"),
                 bg=self.colors["bg_card"], 
                 fg=self.colors["text_main"]).pack(anchor="w", pady=(0, int(20 * self.font_scale)))

        # Section: Parameters
        self._create_section_label(left, "PARAMETRELER")
        self._apple_input(left, "Kaynak Düğüm (S)", self.src)
        self._apple_input(left, "Hedef Düğüm (D)", self.dst)
        self._apple_input(left, "Talep (Bandwidth)", self.dem)

        ttk.Separator(left).pack(fill="x", pady=int(15 * self.font_scale))

        # Section: Weights
        self._create_section_label(left, "OPTİMİZASYON AĞIRLIKLARI")
        self.w1 = self._apple_slider(left, "Gecikme", 0.4)
        self.w2 = self._apple_slider(left, "Güvenilirlik", 0.3)
        self.w3 = self._apple_slider(left, "Kaynak", 0.3)

        ttk.Separator(left).pack(fill="x", pady=int(15 * self.font_scale))

        # Section: Visuals (Collapsible feel)
        self._create_section_label(left, "GÖRSEL AYARLAR")
        self._apple_visual_slider(left, "Node Boyutu", self.var_node_size, 10, 300)
        self._apple_visual_slider(left, "Kenar Saydamlığı", self.var_edge_alpha, 0, 1)
        
        chk = ttk.Checkbutton(left, text="Tüm Kenarları Göster", 
                              variable=self.var_show_edges, command=self.draw_graph)
        chk.pack(anchor="w", pady=(5, 10))

        ttk.Separator(left).pack(fill="x", pady=int(15 * self.font_scale))

        # Section: Algorithm & Action
        self._create_section_label(left, "ALGORİTMA")
        
        combo = ttk.Combobox(left, textvariable=self.algorithm_var, 
                             values=["Genetic Algorithm", "Q-Learning"], state="readonly")
        combo.pack(fill=tk.X, pady=(0, int(15 * self.font_scale)))

        btn = ttk.Button(left, text="ALGORİTMAYI ÇALIŞTIR", 
                         style="Apple.TButton", 
                         command=self.run_selected_algorithm)
        btn.pack(fill=tk.X, pady=(0, int(15 * self.font_scale)))

        # Result Text Area
        tk.Label(left, text="Sonuç Detayı", bg=self.colors["bg_card"],
                 font=("Segoe UI", int(10 * self.font_scale), "bold"),
                 fg=self.colors["text_secondary"]).pack(anchor="w")

        self.result_text = tk.Text(left, height=8, wrap=tk.WORD, 
                                  bg=self.colors["bg_main"], 
                                  relief="flat", 
                                  font=("Consolas", int(9 * self.font_scale)),
                                  padx=10, pady=10)
        self.result_text.pack(fill=tk.X, pady=(5,0))


        # ---------------- RIGHT PANEL (VISUALIZATION) ----------------
        right = tk.Frame(self.root, bg=self.colors["bg_main"], 
                         padx=int(20 * self.font_scale), pady=int(20 * self.font_scale))
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # 1. Metric Cards (Horizontal Layout)
        cards_frame = tk.Frame(right, bg=self.colors["bg_main"])
        cards_frame.pack(fill=tk.X, pady=(0, int(20 * self.font_scale)))
        
        # Using grid or pack side=left for horizontal cards
        self.card_delay = self._create_card(cards_frame, "Ort. Gecikme", "0 ms", self.colors["error"])
        self.card_rel = self._create_card(cards_frame, "Güvenilirlik", "%0", self.colors["success"])
        self.card_res = self._create_card(cards_frame, "Kaynak Maliyeti", "0", self.colors["accent"])

        # 2. Graph Area
        # Adding a frame to give a white border/shadow effect to the graph
        graph_container = tk.Frame(right, bg="white", bd=1, relief="solid")
        # Removing border color actually looks cleaner in modern UI, using padding instead
        graph_container.config(highlightbackground="#D1D1D6", highlightthickness=1, relief="flat")
        graph_container.pack(expand=True, fill=tk.BOTH)
        
        self.fig, self.ax = plt.subplots(figsize=(9, 8), facecolor="white")
        
        # Remove standard matplotlib toolbar if you want clean look, 
        # or keep it. Here we just add the canvas.
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)


    # ================= UI HELPERS =================
    def _create_section_label(self, parent, text):
        tk.Label(parent, text=text, bg=self.colors["bg_card"], 
                 fg=self.colors["text_secondary"],
                 font=("Segoe UI", int(9 * self.font_scale), "bold")).pack(anchor="w", pady=(int(5 * self.font_scale), int(5 * self.font_scale)))

    def _apple_input(self, parent, label_text, variable):
        container = tk.Frame(parent, bg=self.colors["bg_card"])
        container.pack(fill=tk.X, pady=(0, int(10 * self.font_scale)))
        
        tk.Label(container, text=label_text, bg=self.colors["bg_card"], 
                 fg=self.colors["text_main"],
                 font=("Segoe UI", int(11 * self.font_scale))).pack(anchor="w")
        
        # Styled Entry: using a Frame to create a border, then Entry inside
        entry_frame = tk.Frame(container, bg="white", highlightthickness=1, highlightbackground="#D1D1D6")
        entry_frame.pack(fill=tk.X, pady=(2, 0))
        
        if isinstance(variable, tk.IntVar) or isinstance(variable, tk.DoubleVar):
            # For numbers, we still use Entry but could validate. keeping simple.
            ent = tk.Entry(entry_frame, textvariable=variable, relief="flat", 
                           font=("Segoe UI", int(11 * self.font_scale)), bg="white")
            ent.pack(fill=tk.X, padx=5, ipady=int(4 * self.font_scale))
        else:
             ent = tk.Entry(entry_frame, textvariable=variable, relief="flat", 
                           font=("Segoe UI", int(11 * self.font_scale)), bg="white")
             ent.pack(fill=tk.X, padx=5, ipady=int(4 * self.font_scale))

    def _apple_slider(self, parent, label, default_val):
        frame = tk.Frame(parent, bg=self.colors["bg_card"])
        frame.pack(fill=tk.X, pady=(0, int(5 * self.font_scale)))
        
        tk.Label(frame, text=label, bg=self.colors["bg_card"], 
                 font=("Segoe UI", int(10 * self.font_scale)), width=10, anchor="w").pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default_val)
        scale = ttk.Scale(frame, from_=0, to=1, variable=var, orient=tk.HORIZONTAL)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        
        return var

    def _apple_visual_slider(self, parent, label, var, from_val, to_val):
        frame = tk.Frame(parent, bg=self.colors["bg_card"])
        frame.pack(fill=tk.X, pady=(0, int(5 * self.font_scale)))
        
        tk.Label(frame, text=label, bg=self.colors["bg_card"], 
                 font=("Segoe UI", int(10 * self.font_scale))).pack(anchor="w")
        
        scale = ttk.Scale(frame, from_=from_val, to=to_val, variable=var, 
                          orient=tk.HORIZONTAL, command=lambda e: self.draw_graph())
        scale.pack(fill=tk.X, pady=(2, 0))

    def _create_card(self, parent, title, value, color_code):
        # A card is a Frame with a white background and slight padding
        card = tk.Frame(parent, bg=self.colors["bg_card"], 
                        padx=int(20 * self.font_scale), pady=int(15 * self.font_scale))
        
        # Add a subtle border/shadow effect manually via highlight
        card.config(highlightbackground="#E5E5EA", highlightthickness=1)
        
        # Pack horizontally with spacing
        card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=int(5 * self.font_scale))
        
        tk.Label(card, text=title, bg=self.colors["bg_card"], 
                 fg=self.colors["text_secondary"],
                 font=("Segoe UI", int(10 * self.font_scale), "bold")).pack(anchor="center")
        
        lbl_value = tk.Label(card, text=value, bg=self.colors["bg_card"], 
                             fg=color_code,
                             font=("Segoe UI", int(24 * self.font_scale), "bold"))
        lbl_value.pack(anchor="center", pady=(5, 0))
        
        return lbl_value

    # ================= LOGIC & ALGORITHMS =================
    def get_weights(self):
        s = self.w1.get() + self.w2.get() + self.w3.get()
        if s == 0:
            return None
        return {
            "delay": self.w1.get() / s,
            "reliability": self.w2.get() / s,
            "resource": self.w3.get() / s
        }

    def get_demand_mbps(self, src, dst):
        demand_file = r"C:\Users\dkadi\OneDrive - Bartın Üniversitesi\Masaüstü\Bilgisayar Ağları\BSM307_317_Guz2025_TermProject_DemandData.csv"
        try:
            df = pd.read_csv(demand_file, sep=';', decimal=',')
            df.columns = df.columns.str.strip()
            result = df[(df["src"] == src) & (df["dst"] == dst)]
            if result.empty:
                return None
            return result.iloc[0]["demand_mbps"]
        except Exception as e:
            print(f"CSV Okuma Hatası: {e}")
            return None

    def run_selected_algorithm(self):
        s = self.src.get()
        d = self.dst.get()
        
        # Basic validation
        if s not in self.G.nodes or d not in self.G.nodes:
            messagebox.showerror("Geçersiz Düğüm", "Kaynak (S) veya Hedef (D) grafikte bulunamadı.")
            self.current_path = None
            self.SonucYazdir(0, 0, 0, self.algorithm_var.get(), 0, float('inf'), [])
            return

        # Fetch demand
        talep_M = self.get_demand_mbps(s, d)
        if talep_M is None:
            talep = self.dem.get()
            print("Talep Arayüzden Alındı")
        else:
            talep = talep_M
            print(f"Talep CSV'den Alındı: {talep} Mbps")

        algo = self.algorithm_var.get()
        weights = self.get_weights()

        if not weights:
            messagebox.showerror("Hata", "Ağırlıkların toplamı 0 olamaz.")
            return

        # --- EXECUTE ALGORITHM ---
        start = time.perf_counter()
        
        path = None
        cost = float('inf')

        try:
            if algo == "Genetic Algorithm":
                path, cost = run_genetic_algorithm(self.G, s, d, talep, weights)
            else:  # Q-Learning
                path, cost = Q_Learning_run(self.G, s, d, talep, weights)
        except Exception as e:
            messagebox.showerror("Algoritma Hatası", str(e))
            return

        elapsed = time.perf_counter() - start
        self.current_path = path

        # Compute Metrics for the found path
        m = compute_metrics(self.G, path)
        if m is None:
            td, tr, rc = 0, 0, 0
        else:
            td = m["total_delay"]
            tr = m["total_reliability"]
            rc = m["resource_cost"]

        if path is None or len(path) < 2:
            messagebox.showwarning("Yol Bulunamadı", "Geçerli bir yol üretilemedi.")
            self.current_path = None

        self.SonucYazdir(td, tr, rc, algo, elapsed, cost, path)

    def SonucYazdir(self, td, tr, rc, algo, elapsed, cost, path):
        # Update Cards
        self.card_delay.config(text=f"{td:.2f} ms")
        self.card_rel.config(text=f"%{tr * 100:.1f}")
        self.card_res.config(text=f"{rc:.1f}")

        # Update Text Area
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"► Algoritma: {algo}\n")
        self.result_text.insert(tk.END, f"► Süre:      {elapsed:.4f} sn\n")
        self.result_text.insert(tk.END, f"► Fitness:   {cost:.4f}\n")
        self.result_text.insert(tk.END, f"► Yol:       {path}\n")

        # Update Graph Title
        self.ax.set_title(f"{algo} | Süre: {elapsed:.3f} sn", fontsize=12, fontweight='bold', color="#333333")
        self.draw_graph()

    # ================= DRAWING =================
    def draw_graph(self):
        self.ax.clear()
        
        # Draw all nodes
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax,
            node_size=self.var_node_size.get(),
            node_color=self.colors["graph_node"],
            edgecolors=self.colors["graph_edge"],
            linewidths=1.0
        )

        # Draw all edges (optional)
        if self.var_show_edges.get():
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                width=self.var_edge_width.get(),
                alpha=self.var_edge_alpha.get(),
                edge_color=self.colors["graph_edge"]
            )

        # Draw the Path
        if self.current_path:
            # Highlight Source and Dest
            nx.draw_networkx_nodes(
                self.G, self.pos, nodelist=[self.current_path[0], self.current_path[-1]],
                node_size=self.var_node_size.get() * 1.5,
                node_color=self.colors["accent"],
                ax=self.ax
            )
            
            # Highlight Path Edges
            edges = list(zip(self.current_path, self.current_path[1:]))
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=edges,
                edge_color=self.colors["accent"], 
                width=2.5, 
                alpha=1.0,
                ax=self.ax
            )

        self.ax.axis("off")
        self.canvas.draw()

# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = QoSRoutingApp(root)
    root.mainloop()
