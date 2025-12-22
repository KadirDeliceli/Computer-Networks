import random
import networkx as nx
from metrics import compute_metrics, total_cost
from graph_utils import assign_random_edge_attributes


def random_path(G, s, d, talep, max_attempts=200):
    for attempt in range(max_attempts):
        current = s
        path = [s]
        visited = {s}

        for step in range(len(G)):
            if current == d:
                return path

            neighbors = list(G.neighbors(current))
            random.shuffle(neighbors)

            moved = False
            for nb in neighbors:
                if ((nb not in visited) or nb == d) and \
                        G.edges[current, nb]["bandwidth"] >= talep:
                    path.append(nb)
                    visited.add(nb)
                    current = nb
                    moved = True
                    break

            if not moved:
                break

        if current == d:
            return path

    try:
        return nx.shortest_path(G, s, d)
    except nx.NetworkXNoPath:
        return None


def evaluate_population(G, population, weights):
    """
    Popülasyondaki her bireyi (yol) değerlendirir.
    """
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
    """
    Turnuva seçimi (Tournament Selection) yapar.
    """
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def crossover(G, parent1, parent2, s, d):
    """
    İki ebeveyn yoldan çocuk yol üretir (Crossover/Çaprazlama).
    """
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

    if not new_path or new_path[0] != s or new_path[-1] != d:
        return parent1[:]

    for k in range(len(new_path) - 1):
        if not G.has_edge(new_path[k], new_path[k + 1]):
            return parent1[:]

    return new_path


def mutate(G, path, s, d, talep , mutation_rate=0.3):
    """
    Bir yolu mutasyona uğratır.
    """
    if random.random() > mutation_rate or len(path) <= 2:
        return path[:]

    cut = random.randint(0, len(path) - 2)
    prefix = path[: cut + 1]
    node = prefix[-1]

    suffix = random_path(G, node, d , talep)

    if suffix is None or len(suffix) < 2:
        return path[:]

    return prefix + suffix[1:]


def run_genetic_algorithm(G, s, d, talep , weights,
                          pop_size=50, generations=100,
                          mutation_rate=0.3, crossover_rate=0.7):
    """
    Genetik Algoritma (GA) ile S'den D'ye en iyi yolu bulur.
    """
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
        raise RuntimeError("Geçerli yol üretilemedi!")

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
