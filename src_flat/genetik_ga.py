import random
import networkx as nx

from metrics import compute_metrics, total_cost
from graph_utils import assign_random_edge_attributes


# H grafı fallback için sadece + klasik yol
def random_path(G, s, d, talep, max_attempts=200):

    # Belirli sayıda deneme yapılır
    for _ in range(max_attempts):
        current = s
        path = [s]
        visited = {s}

        # Maksimum düğüm sayısı kadar ilerlemeye izin verilir
        for _ in range(len(G)):
            if current == d:
                return path

            # Mevcut düğümün komşuları alınır
            neighbors = list(G.neighbors(current))
            random.shuffle(neighbors)

            moved = False
            for nb in neighbors:
                # Daha önce ziyaret edilen düğümlere geri dönülmemesi için hedef haric
                if nb in visited and nb != d:
                    continue

                # Kenar bilgileri alınır
                data = G.get_edge_data(current, nb, {})
                bw = data.get("bandwidth", None)

                # Bant genişliği talebi karşılanıyorsa ilerlenir
                if bw is not None and bw >= talep:
                    path.append(nb)
                    visited.add(nb)
                    current = nb
                    moved = True
                    break

            # Hiçbir komşuya gidilemediyse bu deneme başarısız
            if not moved:
                break

        # Döngü sonunda hedefe ulaşıldıysa yol döndürülür
        if current == d:
            return path

    # Rastgele yol bulunamazsa:
    # Talep kısıtını sağlayan kenarlardan oluşan alt graf (H) oluşturulur
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    print("H yapısı çalışcak")

    for u, v, data in G.edges(data=True):
        bw = data.get("bandwidth", None)
        # Yalnızca talebi karşılayan kenarları ekliyoruz
        if bw is not None and bw >= talep:
            H.add_edge(u, v, **data)

    # Kaynak veya hedef grafikte yoksa yol yoktur
    if s not in H or d not in H:
        return None

    try:
        # Kısıtı sağlayan en kısa yol bulunur
        print("Kısa yol çalıştı")
        return nx.shortest_path(H, s, d)
    except nx.NetworkXNoPath:
        return None


# Popülasyondaki her yol için maliyet ve fitness hesabını yapıyoruz
def evaluate_population(G, population, weights):
    costs = []
    fitnesses = []

    for path in population:
        metrics = compute_metrics(G, path)

        # Geçersiz yol durumu
        if metrics is None:
            c = float("inf")
            f = 0.0
        else:
            # Ağırlıklı toplam maliyet
            c = total_cost(metrics, weights)
            f = 1.0 / (1.0 + c)

        costs.append(c)
        fitnesses.append(f)

    return costs, fitnesses


# Turnuva seçimi: rastgele k birey arasından en iyisi seçilir
def tournament_selection(population, fitnesses, k=3):

    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


# Çaprazlama (crossover) işlemi
def crossover(G, parent1, parent2, s, d):

    # Başlangıç ve bitiş hariç ortak düğümler bulunur
    common_nodes = list(set(parent1[1:-1]) & set(parent2[1:-1]))

    # Ortak düğüm yoksa parent1 aynen döndürülür
    if not common_nodes:
        return parent1[:]

    # Rastgele bir ortak düğüm seçilir
    c = random.choice(common_nodes)
    i = parent1.index(c)
    j = parent2.index(c)

    # İki parent birleştirilir
    child = parent1[: i + 1] + parent2[j + 1 :]

    # Döngüleri (tekrarlı düğümleri) temizle
    seen = set()
    new_path = []
    for node in child:
        if node in seen:
            continue
        new_path.append(node)
        seen.add(node)

    strategy = random.random()

    # Geçersiz yol durumu (başlangıç / bitiş yanlış)
    if not new_path or new_path[0] != s or new_path[-1] != d:
        if strategy < 0.3:
            # Daha kısa olan parent tercih edilir
            if (len(parent1) <= len(parent2)):
                return parent1[:]
            else:
                return parent2[:]

        else:
            # Rastgele parent döndürülür
            return random.choice([parent1[:], parent2[:]])

    # Kenar bütünlüğü kontrolü
    for k in range(len(new_path) - 1):
        if not G.has_edge(new_path[k], new_path[k + 1]):
            if strategy < 0.3:
                if (len(parent1) <= len(parent2)):
                    return parent1[:]
                else:
                    return parent2[:]
            else:
                return random.choice([parent1[:], parent2[:]])

    return new_path


# Mutasyon: yolun bir noktasından sonrası yeniden üretilir
def mutate(G, path, s, d, talep , mutation_rate=0.3):

    # Mutasyon gerçekleşmezse veya yol çok kısaysa
    if random.random() > mutation_rate or len(path) <= 2:
        return path[:]

    # Rastgele bir kesme noktası
    cut = random.randint(0, len(path) - 2)
    prefix = path[: cut + 1]
    node = prefix[-1]

    # Kalan yol rastgele yeniden oluşturulur
    suffix = random_path(G, node, d , talep)

    # Geçerli yol yoksa eski yol korunur
    if suffix is None or len(suffix) < 2:
        return path[:]

    return prefix + suffix[1:]


# Genetik algoritmanın ana fonksiyonu
def run_genetic_algorithm(
    G, s, d, talep , weights,
    pop_size=50, generations=100,
    mutation_rate=0.3, crossover_rate=0.7
):

    print(pop_size , generations)

    # Kaynak ve hedef arasında yol yoksa fallback kenar ekliyoruz
    if not nx.has_path(G, s, d):
        if not G.has_edge(s, d):
            G.add_edge(s, d)
            assign_random_edge_attributes(G, s, d)

    # Başlangıç popülasyonu oluşturuluruyoruz
    population = []
    for _ in range(pop_size * 5):
        p = random_path(G, s, d , talep)
        if p is not None:
            population.append(p)
        if len(population) >= pop_size:
            break

    # Hiç yol bulunamazsa
    if not population:
        return None, float("inf")

    best_path = None
    best_cost = float("inf")

    # Nesiller boyunca evrim
    for gen in range(generations):
        costs, fitnesses = evaluate_population(G, population, weights)

        # En iyi çözüm güncellenir
        for pth, c in zip(population, costs):
            if c < best_cost:
                best_cost = c
                best_path = pth[:]

        new_population = []

        # Elitizm: en iyi birey doğrudan korunur
        new_population.append(best_path[:])

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=3)
            parent2 = tournament_selection(population, fitnesses, k=3)

            # Çaprazlama
            if random.random() < crossover_rate:
                child = crossover(G, parent1, parent2, s, d)
            else:
                child = parent1[:]

            # Çift mutasyon uygulanır - çeşitliliği artırmak için
            child = mutate(G, child, s, d, talep ,mutation_rate=mutation_rate)
            child = mutate(G, child, s, d, talep ,mutation_rate=mutation_rate)

            new_population.append(child)

        population = new_population

    return best_path, best_cost
