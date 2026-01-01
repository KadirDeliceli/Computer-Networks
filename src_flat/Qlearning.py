import math
import networkx as nx
from collections import defaultdict
from metrics import compute_metrics, total_cost
import random


def Q_Learning_run(G, source, destination, demand, weights):
    # Q-Learning hiperparametreleri: Ogrenme orani, gelecek odul katsayisi ve adim siniri
    episodes = 2500
    max_steps = 70
    alpha = 0.12
    gamma = 0.95

    # Kesif (exploration) parametreleri: Baslangicta rastgele hareket et, zamanla azalt
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    # Guvenilirlik (reliability) degerleri carpimsal oldugu icin (0-1 arasi),
    # bunlari toplamsal maliyete cevirmek adina logaritma donusumu yapiyoruz.
    # Kenarlar icin donusum:
    for u, v, data in G.edges(data=True):
        rel = max(min(data["reliability"], 1.0), 0.01)
        data["rel_cost"] = -math.log(rel)

    # Dugumler icin donusum:
    for n, data in G.nodes(data=True):
        rel = max(min(data["reliability"], 1.0), 0.01)
        data["rel_cost"] = -math.log(rel)

    # Kaynak dugumun guvenilirlik maliyetini ayrica tutalim
    source_rel_cost = G.nodes[source]["rel_cost"]

    # Q-Tablosunu baslatiyoruz (state-action degerlerini tutar)
    Q = defaultdict(dict)

    # Q degerini okumak icin yardimci fonksiyon
    def get_q(s, a):
        return Q[s].get(a, 0.0)

    # Q degerini guncellemek icin yardimci fonksiyon
    def set_q(s, a, v):
        Q[s][a] = v

    # Bir dugumden gidilebilecek gecerli komsulari bulur.
    # Sadece bant genisligi talebi karsilayan kenarlar dikkate alinir.
    def neighbors(n):
        acts = []
        for v in G.neighbors(n):
            bw = G[n][v]["bandwidth"]
            if bw > 0 and (demand <= 0 or bw >= demand):
                acts.append(v)
        return acts

    # Mevcut durumda en yuksek Q degerine sahip aksiyonu (gidilecek dugumu) secer
    def best_action(s, n):
        acts = neighbors(n)
        if not acts:
            return None

        best_q = -math.inf
        best_a = None
        for a in acts:
            q = get_q(s, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    # Rastgele bir komsu secer (Kesif amaciyla kullanilir)
    def explore_action(n):
        acts = neighbors(n)
        if not acts:
            return None
        return random.choice(acts)

    # Egitim dongusu (belirlenen bolum sayisi kadar calisir)
    for ep in range(episodes):
        current = source
        step = 0
        done = False
        source_rel_added = False

        # Hedefe ulasana veya adim siniri dolana kadar dongu
        while not done:
            step += 1
            state_key = (current, demand) # Durum tanimi: (bulunulan dugum, talep)

            # Epsilon-Greedy stratejisi: Epsilon olasilikla rastgele git, yoksa en iyiyi sec
            if random.random() < epsilon:
                nxt = explore_action(current)
            else:
                nxt = best_action(state_key, current)

            # Gidecek yer yoksa donguyu kir (cikmaz sokak)
            if nxt is None:
                break

            # Kenar ve dugum verilerini al
            edge = G[current][nxt]
            node = G.nodes[nxt]

            # Gecikme maliyetini hesapla (Kenar gecikmesi + dugum isleme gecikmesi)
            delay_cost = edge["delay"]
            if nxt != source and nxt != destination:
                delay_cost += node["processing_delay"] * 0.1

            # Guvenilirlik ve kaynak kullanim maliyetlerini hesapla
            reliability_cost = edge["rel_cost"] + node["rel_cost"]
            resource_cost = 1000.0 / max(edge["bandwidth"], 1.0)

            # Toplam maliyeti agirliklara gore hesapla
            cost = (
                weights["delay"] * delay_cost
                + weights["reliability"] * reliability_cost
                + weights["resource"] * resource_cost
            )

            # Odul (Reward), maliyetin negatifi olarak tanimlanir (minimize etmek istedigimiz icin)
            r = -cost

            # Ilk adimda kaynak dugumun guvenilirlik maliyetini de dus
            if not source_rel_added:
                r -= source_rel_cost
                source_rel_added = True

            # Bant genisligi yetersizse ceza ver (filtreye ragmen ek kontrol)
            if edge["bandwidth"] < demand:
                r -= 2.0

            # Hedefe ulasildiysa buyuk odul ver ve bitir
            if nxt == destination:
                r += 5.0
                done = True

            # Adim siniri asildiysa ceza ver ve bitir
            if step >= max_steps:
                r -= 5.0
                done = True

            # Gelecek durum icin maksimum beklenen odulu (max Q) hesapla
            next_state = (nxt, demand)
            future = 0.0
            if not done:
                future = max(
                    (get_q(next_state, a) for a in neighbors(nxt)),
                    default=0.0
                )

            # Bellman denklemi ile Q degerini guncelle
            old_q = get_q(state_key, nxt)
            set_q(state_key, nxt, old_q + alpha * (r + gamma * future - old_q))

            # Konumu guncelle
            current = nxt

        # Her bolum sonunda epsilon degerini azalt (kesifi azalt, somuruyu artir)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Egitim bittikten sonra ogrenilen Q tablosunu kullanarak en iyi yolu cikar
    path = [source]
    current = source
    visited = {source} # Donguleri engellemek icin ziyaret edilenleri tut

    for _ in range(max_steps):
        if current == destination:
            break

        state_key = (current, demand)
        # Sadece en iyi aksiyonlari takip et (artik kesif yok)
        nxt = best_action(state_key, current)

        # Yol tikandiysa veya donguye girdiyse dur
        if nxt is None or nxt in visited:
            break

        path.append(nxt)
        visited.add(nxt)
        current = nxt

    # Eger hedefe ulasilamadiysa basarisiz don
    if path[-1] != destination:
        return None, math.inf

    # Yol metriklerini ve toplam maliyeti hesapla
    metrics = compute_metrics(G, path)
    if metrics is None:
        return None, math.inf

    cost = total_cost(metrics, weights)
    return path, cost

Bunlar nasıl sence
