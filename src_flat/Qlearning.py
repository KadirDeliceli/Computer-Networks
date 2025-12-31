import math
import networkx as nx
from collections import defaultdict
from metrics import compute_metrics , total_cost
import random


def Q_Learning_run(G, source, destination, demand, weights):
    episodes = 2500
    max_steps = 70
    alpha = 0.12
    gamma = 0.95

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    for u, v, data in G.edges(data=True):
        rel = max(min(data["reliability"], 1.0), 0.01)
        data["rel_cost"] = -math.log(rel)

    for n, data in G.nodes(data=True):
        rel = max(min(data["reliability"], 1.0), 0.01)
        data["rel_cost"] = -math.log(rel)

    source_rel_cost = G.nodes[source]["rel_cost"]


    Q = defaultdict(dict)

    def get_q(s, a):
        return Q[s].get(a, 0.0)

    def set_q(s, a, v):
        Q[s][a] = v


    neighbor_cache = {}

    def neighbors(n):
        if n in neighbor_cache:
            return neighbor_cache[n]

        acts = []
        for v in G.neighbors(n):
            bw = G[n][v]["bandwidth"]
            if bw > 0 and (demand <= 0 or bw >= demand):
                acts.append(v)

        neighbor_cache[n] = acts
        return acts

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

    def explore_action(n, episode_idx, step_idx):
        acts = neighbors(n)
        if not acts:
            return None
        return random.choice(acts)

    for ep in range(episodes):
        current = source
        step = 0
        done = False
        source_rel_added = False

        while not done:
            step += 1
            state_key = (current, demand)

            if random.random() < epsilon:
                nxt = explore_action(current, ep, step)
            else:
                nxt = best_action(state_key, current)
            if nxt is None:
                break

            edge = G[current][nxt]
            node = G.nodes[nxt]

            delay_cost = edge["delay"]
            if nxt != source and nxt != destination:
                delay_cost += node["processing_delay"] * 0.1

            reliability_cost = edge["rel_cost"] + node["rel_cost"]
            resource_cost = 1000.0 / max(edge["bandwidth"], 1.0)

            cost = weights["delay"] * delay_cost + weights["reliability"] * reliability_cost + weights["resource"] * resource_cost

            r = -cost

            if not source_rel_added:
                r -= source_rel_cost
                source_rel_added = True

            if edge["bandwidth"] < demand:
                r -= 2.0

            if nxt == destination:
                r += 5.0
                done = True

            if step >= max_steps:
                r -= 5.0
                done = True

            next_state = (nxt, demand)
            future = 0.0
            if not done:
                future = max(
                    (get_q(next_state, a) for a in neighbors(nxt)),
                    default=0.0
                )

            old_q = get_q(state_key, nxt)
            set_q(state_key, nxt, old_q + alpha * (r + gamma * future - old_q))

            current = nxt

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    path = [source]
    current = source
    visited = {source}

    for _ in range(max_steps):
        if current == destination:
            break

        state_key = (current, demand)
        nxt = best_action(state_key, current)

        if nxt is None or nxt in visited:
            break

        path.append(nxt)
        visited.add(nxt)
        current = nxt

    if path[-1] != destination:
        return None, math.inf

    metrics = compute_metrics(G, path)
    if metrics is None:
        return None, math.inf

    cost = total_cost(metrics, weights)
    return path, cost
