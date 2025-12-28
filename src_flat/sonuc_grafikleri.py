import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graph_utils import create_random_graph
from genetik_ga import run_genetic_algorithm
import Qlearning as ql
from metrics import compute_metrics

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


RANDOM_SEED = 42
ORNEK_SAYISI = 10         
AGIRLIKLAR = {"delay": 0.4, "reliability": 0.3, "resource": 0.3}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


BU_DOSYA = os.path.dirname(os.path.abspath(__file__))
PROJE_KOK = os.path.abspath(os.path.join(BU_DOSYA, ".."))
DATA = os.path.join(PROJE_KOK, "data")

EDGE = os.path.join(DATA, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
NODE = os.path.join(DATA, "BSM307_317_Guz2025_TermProject_NodeData.csv")
DEMAND = os.path.join(DATA, "BSM307_317_Guz2025_TermProject_DemandData.csv")


def pareto_maske(noktalar: np.ndarray) -> np.ndarray:
    # Minimization: (Delay, ReliabilityCost, TotalCost) hepsi küçük daha iyi
    n = noktalar.shape[0]
    maske = np.ones(n, dtype=bool)
    for i in range(n):
        if not maske[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j, i'yi domine ediyorsa i pareto değil
            if np.all(noktalar[j] <= noktalar[i]) and np.any(noktalar[j] < noktalar[i]):
                maske[i] = False
                break
    return maske

G = create_random_graph(
    250, 0.03,
    edge_file=EDGE,
    node_file=NODE,
    demand_file=DEMAND
)

df_demand = pd.read_csv(DEMAND, sep=";", decimal=",")
df_demand.columns = df_demand.columns.str.strip()

ornekler = df_demand.sample(ORNEK_SAYISI, random_state=RANDOM_SEED)


sonuclar = {
    "GA": {"delay": [], "rel": [], "res": [], "cost": []},
    "Q-Learning": {"delay": [], "rel": [], "res": [], "cost": []}
}

for _, row in ornekler.iterrows():
    s, d, b = int(row["src"]), int(row["dst"]), float(row["demand_mbps"])

    # --- GA ---
    yol_ga, maliyet_ga = run_genetic_algorithm(G, s, d, b, AGIRLIKLAR)
    if yol_ga:
        m = compute_metrics(G, yol_ga)
        sonuclar["GA"]["delay"].append(m["total_delay"])
        sonuclar["GA"]["rel"].append(m["reliability_cost"])
        sonuclar["GA"]["res"].append(m["resource_cost"])
        sonuclar["GA"]["cost"].append(maliyet_ga)

    # --- Q-Learning ---
    yol_ql, maliyet_ql = ql.Q_Learning_run(G, s, d, b, AGIRLIKLAR)
    if yol_ql:
        m = compute_metrics(G, yol_ql)
        sonuclar["Q-Learning"]["delay"].append(m["total_delay"])
        sonuclar["Q-Learning"]["rel"].append(m["reliability_cost"])
        sonuclar["Q-Learning"]["res"].append(m["resource_cost"])
        sonuclar["Q-Learning"]["cost"].append(maliyet_ql)


# -----------------------------
# ortalamalar 3 bar grafik
# -----------------------------
algoritmalar = ["GA", "Q-Learning"]

ortalama_delay = [np.mean(sonuclar[a]["delay"]) for a in algoritmalar]
ortalama_rel = [np.mean(sonuclar[a]["rel"]) for a in algoritmalar]
ortalama_cost = [np.mean(sonuclar[a]["cost"]) for a in algoritmalar]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(algoritmalar, ortalama_delay)
plt.title("Ortalama Toplam Gecikme")
plt.ylabel("Gecikme")

plt.subplot(1, 3, 2)
plt.bar(algoritmalar, ortalama_rel)
plt.title("Ortalama Güvenilirlik Maliyeti")
plt.ylabel("Güvenilirlik Maliyeti")

plt.subplot(1, 3, 3)
plt.bar(algoritmalar, ortalama_cost)
plt.title("Ortalama Toplam Maliyet")
plt.ylabel("Toplam Maliyet")

plt.tight_layout()


# -----------------------------
# boxplot dağılım
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.boxplot([sonuclar["GA"]["delay"], sonuclar["Q-Learning"]["delay"]],
            labels=["GA", "Q-Learning"], showfliers=True)
plt.title("Toplam Gecikme Dağılımı (Boxplot)")
plt.ylabel("Gecikme")

plt.subplot(1, 3, 2)
plt.boxplot([sonuclar["GA"]["rel"], sonuclar["Q-Learning"]["rel"]],
            labels=["GA", "Q-Learning"], showfliers=True)
plt.title("Güvenilirlik Maliyeti Dağılımı (Boxplot)")
plt.ylabel("Güvenilirlik Maliyeti")

plt.subplot(1, 3, 3)
plt.boxplot([sonuclar["GA"]["cost"], sonuclar["Q-Learning"]["cost"]],
            labels=["GA", "Q-Learning"], showfliers=True)
plt.title("Toplam Maliyet Dağılımı (Boxplot)")
plt.ylabel("Toplam Maliyet")

plt.tight_layout()


# -----------------------------
# 3d poreto
# -----------------------------
ga_noktalar = np.column_stack([sonuclar["GA"]["delay"], sonuclar["GA"]["rel"], sonuclar["GA"]["cost"]]) \
    if len(sonuclar["GA"]["delay"]) else np.empty((0, 3))
ql_noktalar = np.column_stack([sonuclar["Q-Learning"]["delay"], sonuclar["Q-Learning"]["rel"], sonuclar["Q-Learning"]["cost"]]) \
    if len(sonuclar["Q-Learning"]["delay"]) else np.empty((0, 3))

tum_noktalar = np.vstack([ga_noktalar, ql_noktalar]) if (len(ga_noktalar) + len(ql_noktalar)) else np.empty((0, 3))

if tum_noktalar.shape[0] > 0:
    pareto_mask = pareto_maske(tum_noktalar)
    pareto_noktalar = tum_noktalar[pareto_mask]

   
    x_min, x_max = np.nanmin(tum_noktalar[:, 0]), np.nanmax(tum_noktalar[:, 0])
    y_min, y_max = np.nanmin(tum_noktalar[:, 1]), np.nanmax(tum_noktalar[:, 1])
    z_min, z_max = np.nanmin(tum_noktalar[:, 2]), np.nanmax(tum_noktalar[:, 2])

    fig = plt.figure(figsize=(12, 9))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    if ga_noktalar.shape[0] > 0:
        ax3d.scatter(ga_noktalar[:, 0], ga_noktalar[:, 1], ga_noktalar[:, 2],
                     marker="o", alpha=0.65, label="GA")
    if ql_noktalar.shape[0] > 0:
        ax3d.scatter(ql_noktalar[:, 0], ql_noktalar[:, 1], ql_noktalar[:, 2],
                     marker="^", alpha=0.65, label="Q-Learning")

    ax3d.scatter(pareto_noktalar[:, 0], pareto_noktalar[:, 1], pareto_noktalar[:, 2],
                 marker="X", s=120, edgecolors="black", linewidths=1.0,
                 label="Pareto-Optimal")

    ax3d.set_title("3D Pareto (min): Gecikme - Güvenilirlik - Toplam Maliyet")
    ax3d.set_xlabel("Toplam Gecikme", labelpad=10)
    ax3d.set_ylabel("Güvenilirlik Maliyeti", labelpad=10)
    ax3d.set_zlabel("Toplam Maliyet", labelpad=10)
    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_min, z_max)

    ax3d.view_init(elev=20, azim=-60)
    ax3d.legend()

    if ga_noktalar.shape[0] > 0:
        ax_xy.scatter(ga_noktalar[:, 0], ga_noktalar[:, 1], marker="o", alpha=0.6, label="GA")
    if ql_noktalar.shape[0] > 0:
        ax_xy.scatter(ql_noktalar[:, 0], ql_noktalar[:, 1], marker="^", alpha=0.6, label="Q-Learning")
    ax_xy.scatter(pareto_noktalar[:, 0], pareto_noktalar[:, 1], marker="X", s=90,
                  edgecolors="black", linewidths=1.0, label="Pareto")
    ax_xy.set_title("İzdüşüm (XY): Gecikme vs Güvenilirlik")
    ax_xy.set_xlabel("Toplam Gecikme")
    ax_xy.set_ylabel("Güvenilirlik Maliyeti")
    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.legend()

    if ga_noktalar.shape[0] > 0:
        ax_xz.scatter(ga_noktalar[:, 0], ga_noktalar[:, 2], marker="o", alpha=0.6)
    if ql_noktalar.shape[0] > 0:
        ax_xz.scatter(ql_noktalar[:, 0], ql_noktalar[:, 2], marker="^", alpha=0.6)
    ax_xz.scatter(pareto_noktalar[:, 0], pareto_noktalar[:, 2], marker="X", s=90,
                  edgecolors="black", linewidths=1.0)
    ax_xz.set_title("İzdüşüm (XZ): Gecikme vs Toplam Maliyet")
    ax_xz.set_xlabel("Toplam Gecikme")
    ax_xz.set_ylabel("Toplam Maliyet")
    ax_xz.set_xlim(x_min, x_max)
    ax_xz.set_ylim(z_min, z_max)

    if ga_noktalar.shape[0] > 0:
        ax_yz.scatter(ga_noktalar[:, 1], ga_noktalar[:, 2], marker="o", alpha=0.6)
    if ql_noktalar.shape[0] > 0:
        ax_yz.scatter(ql_noktalar[:, 1], ql_noktalar[:, 2], marker="^", alpha=0.6)
    ax_yz.scatter(pareto_noktalar[:, 1], pareto_noktalar[:, 2], marker="X", s=90,
                  edgecolors="black", linewidths=1.0)
    ax_yz.set_title("İzdüşüm (YZ): Güvenilirlik vs Toplam Maliyet")
    ax_yz.set_xlabel("Güvenilirlik Maliyeti")
    ax_yz.set_ylabel("Toplam Maliyet")
    ax_yz.set_xlim(y_min, y_max)
    ax_yz.set_ylim(z_min, z_max)

    plt.tight_layout()


plt.show()
