import random
import networkx as nx

from graph_and_metriks.metrics import compute_metrics, total_cost
from graph_and_metriks.graph_utils import assign_random_edge_attributes


# H grafı fallback için sadece + klasik yol
def random_path(G, s, d, talep, max_attempts=200):
    for _ in range(max_attempts):
        current = s
        path = [s]
        visited = {s}

        for _ in range(len(G)):
            if current == d:
                return path

            neighbors = list(G.neighbors(current))
            random.shuffle(neighbors)

            moved = False
            for nb in neighbors:
                if nb in visited and nb != d:
                    continue

                data = G.get_edge_data(current, nb, {})
                bw = data.get("bandwidth", None)

                if bw is not None and bw >= talep:
                    path.append(nb)
                    visited.add(nb)
                    current = nb
                    moved = True
                    break

            if not moved:
                break

        if current == d:
            return path

    # geçerli yol bulunamazsa kısıtı sağlayan en kısa yolu shortest_path() ile döndür.
    H = G.__class__()
    H.add_nodes_from(G.nodes(data=True))
    print("H yapısı çalışcak")
    for u, v, data in G.edges(data=True):
        bw = data.get("bandwidth", None)
        if bw is not None and bw >= talep:
            H.add_edge(u, v, **data)
    
    if s not in H or d not in H:
        return None

    try:
        print("Kısa yol çalıştı")
        return nx.shortest_path(H, s, d)
    except nx.NetworkXNoPath:
        return None

def evaluate_population(G, population, weights):
    costs = []
    fitnesses = []

    for path in population:
        metrics = compute_metrics(G, path)

        if metrics is None:
            c = float("inf")
            f = 0.0
        else:
            c = total_cost(metrics, weights)
            f = 1.0 / (1.0 + c)

        costs.append(c)
        fitnesses.append(f)

    return costs, fitnesses

def tournament_selection(population, fitnesses, k=3):

    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def crossover(G, parent1, parent2, s, d):

    common_nodes = list(set(parent1[1:-1]) & set(parent2[1:-1]))

    if not common_nodes:
        return parent1[:]

    c = random.choice(common_nodes)
    i = parent1.index(c)
    j = parent2.index(c)

    child = parent1[: i + 1] + parent2[j + 1 :]

    seen = set()
    new_path = []
    for node in child:
        if node in seen:
            continue
        new_path.append(node)
        seen.add(node)

    strategy = random.random()
    if not new_path or new_path[0] != s or new_path[-1] != d:
        if strategy < 0.3:
            # En iyi parent'ı döndür
            return parent1[:] if len(parent1) <= len(parent2) else parent2[:]
        else:
            # Rastgele parent seç
            return random.choice([parent1[:], parent2[:]])

    for k in range(len(new_path) - 1):
        if not G.has_edge(new_path[k], new_path[k + 1]):
            if strategy < 0.3:
                # En iyi parent'ı döndür
                return parent1[:] if len(parent1) <= len(parent2) else parent2[:]
            else:
                # Rastgele parent seç
                return random.choice([parent1[:], parent2[:]])
    return new_path

def mutate(G, path, s, d, talep , mutation_rate=0.3):

    if random.random() > mutation_rate or len(path) <= 2:
        return path[:]

    cut = random.randint(0, len(path) - 2)
    prefix = path[: cut + 1]
    node = prefix[-1]

    suffix = random_path(G, node, d , talep)

    if suffix is None or len(suffix) < 2:
        return path[:]

    return prefix + suffix[1:]

def run_genetic_algorithm(G, s, d, talep , weights,pop_size=50, generations=100,mutation_rate=0.3, crossover_rate=0.7):

    if not nx.has_path(G, s, d):
        if not G.has_edge(s, d):
            G.add_edge(s, d)
            assign_random_edge_attributes(G, s, d)

    population = []
    for _ in range(pop_size * 5):
        p = random_path(G, s, d , talep)
        if p is not None:
            population.append(p)
        if len(population) >= pop_size:
            break

    if not population:
        return None, float("inf")

    best_path = None
    best_cost = float("inf")

    for gen in range(generations):
        costs, fitnesses = evaluate_population(G, population, weights)

        for pth, c in zip(population, costs):
            if c < best_cost:
                best_cost = c
                best_path = pth[:]

        new_population = []
        new_population.append(best_path[:])

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, k=3)
            parent2 = tournament_selection(population, fitnesses, k=3)

            if random.random() < crossover_rate:
                child = crossover(G, parent1, parent2, s, d)
            else:
                child = parent1[:]

            child = mutate(G, child, s, d, talep ,mutation_rate=mutation_rate)

            new_population.append(child)

        population = new_population

    return best_path, best_cost
