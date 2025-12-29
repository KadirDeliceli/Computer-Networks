import time
import threading
import os
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import pandas as pd

# ==========================================
# IMPORTLAR (Hata Yönetimi ile)
# ==========================================
try:
    from genetik_ga import run_genetic_algorithm
    from graph_utils import create_random_graph
    from metrics import compute_metrics
    from Qlearning import Q_Learning_run
except ImportError as e:
    print(f"Kritik Import Hatası: {e}")
    # Kodun çökmemesi için dummy fonksiyonlar (gerekirse)
    pass


class QoSRoutingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QoS Tabanlı Çok Amaçlı Rotalama")

        # Pencere açıldığında tam ekran yap veya maksimize et
        try:
            self.root.state('zoomed')
        except:
            w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}")

        self.root.minsize(1000, 700)

        # ==========================================
        # 1. TASARIM AYARLARI
        # ==========================================
        self.font_scale = 1.1
        self.colors = {
            "bg_main": "#F2F2F7",  # System Gray 6
            "bg_card": "#FFFFFF",  # Pure White
            "accent": "#007AFF",  # System Blue
            "accent_hover": "#005BB5",
            "text_main": "#1C1C1E",
            "text_secondary": "#8E8E93",
            "success": "#34C759",
            "error": "#FF3B30",
            "border": "#D1D1D6",
            "graph_node": "#800080",
            "graph_edge": "#A52A2A"
        }
        self.root.configure(bg=self.colors["bg_main"])

        # ==========================================
        # 2. VERİ YÜKLEME (DOSYA YOLU GÜVENLİĞİ)
        # ==========================================
        # Kodun çalıştığı klasörü baz alarak data klasörünü bulur
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "..", "data")

        self.edge_file = os.path.join(data_dir, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
        self.node_file = os.path.join(data_dir, "BSM307_317_Guz2025_TermProject_NodeData.csv")
        self.demand_file = os.path.join(data_dir, "BSM307_317_Guz2025_TermProject_DemandData.csv")

        try:
            self.G = create_random_graph(
                250, 0.03,
                edge_file=self.edge_file,
                demand_file=self.demand_file,
                node_file=self.node_file
            )
        except Exception as e:
            print(f"Veri yükleme hatası (Test moduna geçiliyor): {e}")
            self.G = nx.erdos_renyi_graph(50, 0.1)
            for u, v in self.G.edges():
                self.G[u][v]['weight'] = 1
                self.G[u][v]['bandwidth'] = 1000
                self.G[u][v]['delay'] = 5
                self.G[u][v]['reliability'] = 0.99
            for n in self.G.nodes():
                self.G.nodes[n]['processing_delay'] = 1
                self.G.nodes[n]['reliability'] = 0.99

        self.pos = nx.spring_layout(self.G, seed=42)
        self.current_path = None
        self.press = None

        # ==========================================
        # 3. DEĞİŞKENLER
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
        # 4. ARAYÜZ İNŞASI
        # ==========================================
        self.build_responsive_layout()
        self.draw_graph()

    def build_responsive_layout(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                                    bg=self.colors["bg_main"], sashwidth=4, sashrelief=tk.RAISED)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # --- SOL PANEL (Scrollable) ---
        self.left_container = tk.Frame(self.paned, bg=self.colors["bg_card"], width=400)
        self.paned.add(self.left_container, minsize=300)

        self.canvas_scroll = tk.Canvas(self.left_container, bg=self.colors["bg_card"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.canvas_scroll.yview)
        self.scrollable_frame = tk.Frame(self.canvas_scroll, bg=self.colors["bg_card"], padx=20, pady=20)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )

        self.canvas_frame_window = self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.bind("<Configure>", self._on_canvas_configure)
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.bind_mouse_scroll(self.canvas_scroll)
        self.bind_mouse_scroll(self.scrollable_frame)

        self.populate_left_panel(self.scrollable_frame)

        # --- SAĞ PANEL (Grafik & Kartlar) ---
        self.right_container = tk.Frame(self.paned, bg=self.colors["bg_main"], padx=10, pady=10)
        self.paned.add(self.right_container, minsize=500)

        # Kartlar
        cards_frame = tk.Frame(self.right_container, bg=self.colors["bg_main"])
        cards_frame.pack(fill=tk.X, pady=(0, 10))
        cards_frame.columnconfigure(0, weight=1)
        cards_frame.columnconfigure(1, weight=1)
        cards_frame.columnconfigure(2, weight=1)

        self.card_delay = self._create_card(cards_frame, 0, "Gecikme", "0 ms", self.colors["error"])
        self.card_rel = self._create_card(cards_frame, 1, "Güvenilirlik", "%0", self.colors["success"])
        self.card_res = self._create_card(cards_frame, 2, "Kaynak", "0", self.colors["accent"])

        # Grafik Alanı
        graph_container = tk.Frame(self.right_container, bg="white", bd=0, highlightthickness=0)
        graph_container.pack(expand=True, fill=tk.BOTH)

        self.fig, self.ax = plt.subplots(figsize=(5, 4), facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        self.canvas.mpl_connect("scroll_event", self.on_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)

    def _on_canvas_configure(self, event):
        self.canvas_scroll.itemconfig(self.canvas_frame_window, width=event.width)

    def bind_mouse_scroll(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        widget.bind_all("<Button-4>", self._on_mousewheel)
        widget.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        x, y = self.root.winfo_pointerxy()
        widget = self.root.winfo_containing(x, y)
        if str(widget).startswith(str(self.left_container)):
            if event.num == 5 or event.delta == -120:
                self.canvas_scroll.yview_scroll(1, "units")
            elif event.num == 4 or event.delta == 120:
                self.canvas_scroll.yview_scroll(-1, "units")

    # ================= UI İÇERİK =================
    def populate_left_panel(self, parent):
        tk.Label(parent, text="QoS Routing",
                 font=("SF Pro Display", int(24 * self.font_scale), "bold"),
                 bg=self.colors["bg_card"], fg=self.colors["text_main"]).pack(anchor="w", pady=(0, 20))

        self._create_section_label(parent, "PARAMETRELER")
        self._rounded_input(parent, "Kaynak Düğüm (S)", self.src)
        self._rounded_input(parent, "Hedef Düğüm (D)", self.dst)
        self._rounded_input(parent, "Talep (Bandwidth)", self.dem)

        self._separator(parent)

        self._create_section_label(parent, "OPTİMİZASYON AĞIRLIKLARI")
        self.w1 = self._rounded_slider(parent, "Gecikme", 0.4)
        self.w2 = self._rounded_slider(parent, "Güvenilirlik", 0.3)
        self.w3 = self._rounded_slider(parent, "Kaynak", 0.3)

        self._separator(parent)

        self._create_section_label(parent, "GÖRSEL AYARLAR")
        self._rounded_slider(parent, "Node Boyutu", 60, min_val=10, max_val=200,
                             variable=self.var_node_size, command=self.draw_graph)
        self._rounded_slider(parent, "Kenar Kalınlığı", 0.5, min_val=0.1, max_val=5.0,
                             variable=self.var_edge_width, command=self.draw_graph)
        self._rounded_slider(parent, "Kenar Saydamlığı", 0.6, min_val=0.0, max_val=1.0,
                             variable=self.var_edge_alpha, command=self.draw_graph)

        tk.Checkbutton(parent, text="Tüm Kenarları Göster", variable=self.var_show_edges,
                       bg=self.colors["bg_card"], command=self.draw_graph).pack(anchor="w", pady=5)

        self._separator(parent)

        self._create_section_label(parent, "ALGORİTMA")
        ttk.Combobox(parent, textvariable=self.algorithm_var,
                     values=["Genetic Algorithm", "Q-Learning"], state="readonly").pack(fill=tk.X, pady=(0, 15))

        # BUTON
        self.btn_run_canvas = self._rounded_button(parent, "ALGORİTMAYI ÇALIŞTIR", self.run_selected_algorithm)

        tk.Label(parent, text="Sonuç Detayı", bg=self.colors["bg_card"],
                 font=("Segoe UI", int(10 * self.font_scale), "bold"),
                 fg=self.colors["text_secondary"]).pack(anchor="w", pady=(15, 5))

        self.result_text_widget = self._rounded_text_area(parent, height=150)

    # ================= LOGIC & THREADING =================

    def get_weights(self):
        s = self.w1.get() + self.w2.get() + self.w3.get()
        if s == 0: return None
        return {"delay": self.w1.get() / s, "reliability": self.w2.get() / s, "resource": self.w3.get() / s}

    def get_demand_mbps(self, src, dst):
        try:
            df = pd.read_csv(self.demand_file, sep=';', decimal=',')
            df.columns = df.columns.str.strip()
            res = df[(df["src"] == src) & (df["dst"] == dst)]
            return res.iloc[0]["demand_mbps"] if not res.empty else None
        except:
            return None

    def toggle_ui_state(self, is_running):
        """Çalışma durumuna göre UI'ı kilitler/açar"""
        if is_running:
            self.root.config(cursor="watch")
            self.result_text_widget.delete("1.0", tk.END)
            self.result_text_widget.insert(tk.END,
                                           "► Algoritma başlatılıyor...\n► Lütfen bekleyiniz, işlem devam ediyor.\n")
            # Butonu pasif hale getirmek isterseniz buraya ekleyebilirsiniz
        else:
            self.root.config(cursor="")

    def run_selected_algorithm(self):
        """Ana iş parçacığını (UI) dondurmadan algoritmayı başlatır."""
        s, d = self.src.get(), self.dst.get()

        # 1. Kontroller
        if s not in self.G or d not in self.G:
            messagebox.showerror("Hata", f"Geçersiz Düğüm: {s} veya {d} grafikte bulunamadı.")
            return

        talep = self.get_demand_mbps(s, d) or self.dem.get()
        weights = self.get_weights()
        if not weights:
            messagebox.showwarning("Uyarı", "Ağırlıklar toplamı 0 olamaz.")
            return

        algo = self.algorithm_var.get()

        # 2. UI Durumu Güncelle
        self.toggle_ui_state(True)

        # 3. Worker Thread Tanımla
        def worker_thread():
            try:
                start_time = time.perf_counter()
                path = None
                cost = 0

                if algo == "Genetic Algorithm":
                    path, cost = run_genetic_algorithm(self.G, s, d, talep, weights)
                elif algo == "Q-Learning":
                    path, cost = Q_Learning_run(self.G, s, d, talep, weights)

                elapsed = time.perf_counter() - start_time

                # İşlem bitince Ana Thread'e haber ver
                self.root.after(0, lambda: self.on_algorithm_complete(path, cost, elapsed, algo))

            except Exception as e:
                self.root.after(0, lambda: self.on_algorithm_error(str(e)))

        # 4. Thread Başlat
        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()

    def on_algorithm_complete(self, path, cost, elapsed, algo):
        """Algoritma bittiğinde ana thread tarafından çağrılır."""
        self.toggle_ui_state(False)
        self.current_path = path

        if not path:
            messagebox.showinfo("Sonuç", "Kısıtları sağlayan uygun bir yol bulunamadı.")
            self.result_text_widget.insert(tk.END, "► Sonuç: BAŞARISIZ (Yol Yok)\n")
            return

        # Metrikleri hesapla
        try:
            m = compute_metrics(self.G, path)
            td = m["total_delay"]
            tr = m["total_reliability"]
            rc = m["resource_cost"]
        except:
            td, tr, rc = 0, 0, 0

        # Kartları Güncelle
        self.card_delay.config(text=f"{td:.2f} ms")
        self.card_rel.config(text=f"%{tr * 100:.1f}")
        self.card_res.config(text=f"{rc:.1f}")

        # Text Alanını Güncelle
        self.result_text_widget.insert(tk.END, f"► Durum:   TAMAMLANDI\n")
        self.result_text_widget.insert(tk.END, f"► Yöntem:  {algo}\n")
        self.result_text_widget.insert(tk.END, f"► Süre:    {elapsed:.4f} sn\n")
        self.result_text_widget.insert(tk.END, f"► Fitness: {cost:.4f}\n")
        self.result_text_widget.insert(tk.END, f"► Uzunluk: {len(path)} node\n")
        self.result_text_widget.insert(tk.END, f"► Rota:    {path}\n")

        # Grafiği Güncelle
        self.ax.set_title(f"{algo} | Süre: {elapsed:.3f} sn | Fit: {cost:.2f}")
        self.draw_graph()

    def on_algorithm_error(self, error_msg):
        self.toggle_ui_state(False)
        messagebox.showerror("Algoritma Hatası", f"Beklenmeyen bir hata oluştu:\n{error_msg}")

    # ================= GRAFİK ÇİZİMİ =================
    def draw_graph(self):
        self.ax.clear()
        try:
            # Node Çizimi
            nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax,
                                   node_size=self.var_node_size.get(),
                                   node_color=self.colors["graph_node"],
                                   edgecolors=self.colors["graph_edge"])
            # Edge Çizimi
            if self.var_show_edges.get():
                nx.draw_networkx_edges(self.G, self.pos, ax=self.ax,
                                       width=self.var_edge_width.get(),
                                       alpha=self.var_edge_alpha.get(),
                                       edge_color=self.colors["graph_edge"])
            # Yol Varsa Çizimi
            if self.current_path:
                # Başlangıç ve Bitiş Node'larını vurgula
                nx.draw_networkx_nodes(self.G, self.pos,
                                       nodelist=[self.current_path[0], self.current_path[-1]],
                                       node_size=self.var_node_size.get() * 1.5,
                                       node_color=self.colors["accent"],
                                       ax=self.ax)
                # Yolu çiz
                edges_in_path = list(zip(self.current_path, self.current_path[1:]))
                nx.draw_networkx_edges(self.G, self.pos, edgelist=edges_in_path,
                                       edge_color=self.colors["accent"],
                                       width=self.var_edge_width.get() * 3,
                                       ax=self.ax)
        except Exception as e:
            print(f"Çizim hatası: {e}")

        self.ax.axis("off")
        self.canvas.draw()

    # ================= MATPLOTLIB ETKİLEŞİM =================
    def on_zoom(self, event):
        if event.inaxes != self.ax: return
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None: return
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self.press = event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim()

    def on_drag(self, event):
        if self.press is None or event.inaxes != self.ax: return
        x0, y0, xlim0, ylim0 = self.press
        dx = event.x - x0
        dy = event.y - y0
        width, height = self.canvas.get_width_height()
        dx_data = (xlim0[1] - xlim0[0]) * (-dx / width)
        dy_data = (ylim0[1] - ylim0[0]) * (-dy / height)
        self.ax.set_xlim(xlim0[0] + dx_data, xlim0[1] + dx_data)
        self.ax.set_ylim(ylim0[0] + dy_data, ylim0[1] + dy_data)
        self.canvas.draw_idle()

    def on_release(self, event):
        self.press = None
        self.canvas.draw_idle()

    # ================= ÖZEL WIDGET'LAR =================
    def _round_rectangle(self, canvas, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1 + radius, y1, x1 + radius, y1, x2 - radius, y1, x2 - radius, y1,
                  x2, y1, x2, y1 + radius, x2, y1 + radius, x2, y2 - radius, x2, y2 - radius,
                  x2, y2, x2 - radius, y2, x2 - radius, y2, x1 + radius, y2, x1 + radius, y2,
                  x1, y2, x1, y2 - radius, x1, y2 - radius, x1, y1 + radius, x1, y1]
        return canvas.create_polygon(points, **kwargs, smooth=True)

    def _rounded_input(self, parent, label_text, variable):
        container = tk.Frame(parent, bg=self.colors["bg_card"])
        container.pack(fill=tk.X, pady=(0, 12))
        tk.Label(container, text=label_text, bg=self.colors["bg_card"], fg=self.colors["text_main"],
                 font=("Segoe UI", int(11 * self.font_scale))).pack(anchor="w", padx=5)
        h = int(35 * self.font_scale)
        canvas = tk.Canvas(container, height=h, bg=self.colors["bg_card"], bd=0, highlightthickness=0)
        canvas.pack(fill=tk.X, pady=(2, 0))

        def draw(event):
            canvas.delete("all")
            w = event.width
            self._round_rectangle(canvas, 2, 2, w - 2, h - 2, radius=h - 4, outline=self.colors["border"], fill="white",
                                  width=1)

        canvas.bind("<Configure>", draw)
        entry = tk.Entry(canvas, textvariable=variable, bd=0, bg="white", highlightthickness=0,
                         font=("Segoe UI", int(11 * self.font_scale)))
        entry.place(relx=0.05, rely=0.25, relwidth=0.9, relheight=0.5)

    def _rounded_button(self, parent, text, command):
        h = int(45 * self.font_scale)
        canvas = tk.Canvas(parent, height=h, bg=self.colors["bg_card"], bd=0, highlightthickness=0, cursor="hand2")
        canvas.pack(fill=tk.X, pady=10)

        def draw(event):
            canvas.delete("all")
            w = event.width
            self._round_rectangle(canvas, 2, 2, w - 2, h - 2, radius=h - 4, fill=self.colors["accent"], outline="")
            canvas.create_text(w / 2, h / 2, text=text, fill="white",
                               font=("Segoe UI", int(11 * self.font_scale), "bold"))

        canvas.bind("<Configure>", draw)
        canvas.bind("<Button-1>", lambda e: command())
        return canvas

    def _rounded_text_area(self, parent, height=100):
        canvas = tk.Canvas(parent, height=height, bg=self.colors["bg_card"], bd=0, highlightthickness=0)
        canvas.pack(fill=tk.X)

        def draw(event):
            canvas.delete("bg")
            w = event.width
            h = event.height
            self._round_rectangle(canvas, 2, 2, w - 2, h - 2, radius=20, outline=self.colors["border"],
                                  fill=self.colors["bg_main"], tags="bg")

        canvas.bind("<Configure>", draw)
        text_widget = tk.Text(canvas, bg=self.colors["bg_main"], bd=0, highlightthickness=0,
                              font=("Consolas", int(10 * self.font_scale)), wrap=tk.WORD)
        text_widget.place(relx=0.03, rely=0.05, relwidth=0.94, relheight=0.9)
        return text_widget

    def _rounded_slider(self, parent, label_text, default_val, min_val=0.0, max_val=1.0, variable=None, command=None):
        if variable is None: variable = tk.DoubleVar(value=default_val)
        frame = tk.Frame(parent, bg=self.colors["bg_card"])
        frame.pack(fill=tk.X, pady=(0, 10))
        header = tk.Frame(frame, bg=self.colors["bg_card"])
        header.pack(fill=tk.X)
        tk.Label(header, text=label_text, bg=self.colors["bg_card"], fg=self.colors["text_main"],
                 font=("Segoe UI", int(10 * self.font_scale))).pack(side=tk.LEFT)
        val_lbl = tk.Label(header, text=f"{variable.get():.2f}", bg=self.colors["bg_card"],
                           fg=self.colors["text_secondary"], font=("Segoe UI", int(10 * self.font_scale)))
        val_lbl.pack(side=tk.RIGHT)
        h = 20
        c = tk.Canvas(frame, height=h, bg=self.colors["bg_card"], bd=0, highlightthickness=0, cursor="hand2")
        c.pack(fill=tk.X, pady=(5, 0))

        def update_slider(event=None):
            w = c.winfo_width()
            val = variable.get()
            ratio = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
            x_pos = 10 + ratio * (w - 20)
            c.delete("all")
            c.create_line(10, h / 2, w - 10, h / 2, fill="#D1D1D6", width=4, capstyle=tk.ROUND)
            c.create_line(10, h / 2, x_pos, h / 2, fill="#007AFF", width=4, capstyle=tk.ROUND)
            c.create_oval(x_pos - 8, (h / 2) - 8, x_pos + 8, (h / 2) + 8, fill="#FFFFFF", outline="#D1D1D6", width=1)
            val_lbl.config(text=f"{val:.2f}" if max_val <= 1 else f"{int(val)}")

        def on_click(event):
            w = c.winfo_width()
            x = max(10, min(event.x, w - 10))
            ratio = (x - 10) / (w - 20)
            variable.set(max(min_val, min(max_val, min_val + ratio * (max_val - min_val))))
            update_slider()
            if command: command()

        c.bind("<Configure>", lambda e: update_slider())
        c.bind("<Button-1>", on_click)
        c.bind("<B1-Motion>", on_click)
        parent.after(100, update_slider)
        return variable

    def _separator(self, parent):
        ttk.Separator(parent).pack(fill="x", pady=15)

    def _create_section_label(self, parent, text):
        tk.Label(parent, text=text, bg=self.colors["bg_card"],
                 fg=self.colors["text_secondary"],
                 font=("Segoe UI", int(9 * self.font_scale), "bold")).pack(anchor="w", pady=(5, 5))

    def _create_card(self, parent, col, title, value, color_code):
        card = tk.Frame(parent, bg=self.colors["bg_card"],
                        padx=int(10 * self.font_scale), pady=int(10 * self.font_scale))
        card.config(highlightbackground="#E5E5EA", highlightthickness=1)
        card.grid(row=0, column=col, padx=5, sticky="ew")

        tk.Label(card, text=title, bg=self.colors["bg_card"], fg=self.colors["text_secondary"],
                 font=("Segoe UI", int(10 * self.font_scale), "bold")).pack()
        lbl = tk.Label(card, text=value, bg=self.colors["bg_card"], fg=color_code,
                       font=("Segoe UI", int(20 * self.font_scale), "bold"))
        lbl.pack(pady=(5, 0))
        return lbl


if __name__ == "__main__":
    root = tk.Tk()
    app = QoSRoutingApp(root)
    root.mainloop()
