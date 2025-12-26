import random
import math
import networkx as nx

from graph_and_metriks.metrics import compute_metrics, total_cost




def normalize_weights(weights):
    d = weights.get("delay", 0.0)
    r = weights.get("reliability", 0.0)
    s = weights.get("resource", 0.0)
    t = d + r + s
    if t == 0:
        return 1/3, 1/3, 1/3
    return d/t, r/t, s/t


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
    w_delay, w_rel, w_res = normalize_weights(weights)

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

    congestion_penalty = 0.0
    if demand > 0 and bandwidth < demand:
        congestion_penalty = 3.0 * ((demand - bandwidth) / demand) * 2

    slide_booster = 3.4

    total = (w_delay* slide_booster) * delay_cost + (w_rel * slide_booster) * reliability_cost + (w_res * slide_booster) * resource_cost + congestion_penalty 

    return total, delay_cost, reliability_cost, resource_cost, bandwidth


def Q_Learning_run(G, source, destination, demand, weights):
    episodes = 2200
    max_steps = 70
    alpha = 0.12
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    Q = {}

    def get_q(s, a):
        return Q.get(s, {}).get(a, 0.0)

    def set_q(s, a, v):
        if s not in Q:
            Q[s] = {}
        Q[s][a] = v

    def neighbors(n):
        return list(G.neighbors(n))

    def best_action(s, n):
        acts = neighbors(n)
        if not acts:
            return None
        return max(acts, key=lambda a: get_q(s, a))

    def choose_action(s, n):
        if random.random() < epsilon:
            return random.choice(neighbors(n))
        return best_action(s, n)

    global state, action, reward

    priority_mode = compute_priority_mode(weights)

    src_rel = float(G.nodes[source]["reliability"])
    src_rel = max(min(src_rel, 1.0), 0.01)
    source_rel_cost = -math.log(src_rel)

    for _ in range(episodes):
        current = source
        step = 0
        done = False
        source_rel_added = False

        while not done:
            step += 1

            state_key = (current, priority_mode)
            state = state_key

            nxt = choose_action(state_key, current)
            if nxt is None:
                break

            action = nxt

            cost, d_cost, r_cost, res_cost, bw = step_cost(
                G, current, nxt, source, destination, demand, weights
            )

            r = -cost

            if not source_rel_added:
                r -= source_rel_cost
                source_rel_added = True

            if priority_mode == 0:
                r -= 3.0 * d_cost
            elif priority_mode == 1:
                r -= 3.0 * r_cost
            else:
                r -= 3.0 * res_cost

            if bw < demand:
                r -= 2.0

            if nxt == destination:
                r += 1.0 + priority_mode
                done = True

            if step >= max_steps:
                r -= 5.0
                done = True

            reward = r

            next_state = (nxt, priority_mode)
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

    for _ in range(max_steps):
        if current == destination:
            break
        state_key = (current, priority_mode)
        nxt = best_action(state_key, current)
        if nxt is None:
            break
        path.append(nxt)
        current = nxt

    metrics = compute_metrics(G, path)
    cost = total_cost(metrics, weights)

    return path, cost
