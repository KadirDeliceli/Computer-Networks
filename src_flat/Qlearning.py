import math
import networkx as nx

from metrics import compute_metrics, total_cost


def compute_priority_mode(weights):
    d = weights["delay"]
    r = weights["reliability"]
    s = weights["resource"]
    if d >= r and d >= s:
        return 0
    elif r >= d and r >= s:
        return 1
    else:
        return 2


def step_cost(G, u, v, source, destination, demand, weights):
    w_delay = weights["delay"]
    w_rel = weights["reliability"]
    w_res = weights["resource"]

    edge = G[u][v]
    node = G.nodes[v]

    bandwidth = edge["bandwidth"]
    delay = edge["delay"]
    link_rel = edge["reliability"]
    node_rel = node["reliability"]
    proc = node["processing_delay"]

    delay_cost = delay
    if v != source and v != destination:
        delay_cost += proc

    link_rel = max(min(link_rel, 1.0), 0.01)
    node_rel = max(min(node_rel, 1.0), 0.01)

    reliability_cost = -math.log(link_rel) - math.log(node_rel)
    resource_cost = 1000.0 / max(bandwidth, 1.0)

    total = (
        w_delay * delay_cost
        + w_rel * reliability_cost
        + w_res * resource_cost
    )

    return total, delay_cost, reliability_cost, resource_cost, bandwidth


def Q_Learning_run(G, source, destination, demand, weights):
    episodes = 2500
    max_steps = 70
    alpha = 0.12
    gamma = 0.95

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    PRIORITY_PENALTY = 1.0

    Q = {}

    def get_q(s, a):
        return Q.get(s, {}).get(a, 0.0)

    def set_q(s, a, v):
        if s not in Q:
            Q[s] = {}
        Q[s][a] = v

    def neighbors(n):
        acts = []
        for v in G.neighbors(n):
            bw = G[n][v]["bandwidth"]
            if bw <= 0:
                continue
            if demand > 0 and bw < demand:
                continue
            acts.append(v)
        return acts

    def best_action(s, n):
        acts = neighbors(n)
        if not acts:
            return None
        qs = [(a, get_q(s, a)) for a in acts]
        max_q = max(qs, key=lambda x: x[1])[1]
        best = [a for a, q in qs if q == max_q]
        return min(best)

    def explore_action(s, n, episode_idx, step_idx):
        acts = neighbors(n)
        if not acts:
            return None
        i = (episode_idx + step_idx) % len(acts)
        return acts[i]

    priority_mode = compute_priority_mode(weights)

    src_rel = float(G.nodes[source]["reliability"])
    src_rel = max(min(src_rel, 1.0), 0.01)
    source_rel_cost = -math.log(src_rel)

    for ep in range(episodes):
        current = source
        step = 0
        done = False
        source_rel_added = False

        while not done:
            step += 1
            state_key = (current, priority_mode, demand)

            if step <= int(max_steps * epsilon):
                nxt = explore_action(state_key, current, ep, step)
            else:
                nxt = best_action(state_key, current)

            if nxt is None:
                break

            cost, d_cost, r_cost, res_cost, bw = step_cost(
                G, current, nxt, source, destination, demand, weights
            )

            r = -cost

            if not source_rel_added:
                r -= source_rel_cost
                source_rel_added = True

            if priority_mode == 0:
                r -= PRIORITY_PENALTY * d_cost
            elif priority_mode == 1:
                r -= PRIORITY_PENALTY * r_cost
            else:
                r -= PRIORITY_PENALTY * res_cost

            if bw < demand:
                r -= 2.0

            if nxt == destination:
                r += 5.0
                done = True

            if step >= max_steps:
                r -= 5.0
                done = True

            next_state = (nxt, priority_mode, demand)
            future = 0.0 if done else max(
                (get_q(next_state, a) for a in neighbors(nxt)),
                default=0.0
            )

            old_q = get_q(state_key, nxt)
            set_q(state_key, nxt, old_q + alpha * (r + gamma * future - old_q))

            current = nxt

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    path = [source]
    current = source
    visited = set([source])

    for _ in range(max_steps):
        if current == destination:
            break

        state_key = (current, priority_mode, demand)
        nxt = best_action(state_key, current)

        if nxt is None:
            break

        if nxt in visited:
            break

        path.append(nxt)
        visited.add(nxt)
        current = nxt

    if path[-1] != destination:
        return None, float("inf")

    metrics = compute_metrics(G, path)

    if metrics is None:
        return None, float("inf")

    cost = total_cost(metrics, weights)
    return path, cost
