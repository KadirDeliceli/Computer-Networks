import time
import random
import numpy as np
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

parent_dir = os.path.dirname(current_dir)
edge_path = os.path.join(parent_dir, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
node_path = os.path.join(parent_dir, "BSM307_317_Guz2025_TermProject_NodeData.csv")
demand_path = os.path.join(parent_dir, "BSM307_317_Guz2025_TermProject_DemandData.csv")

from qlearning.qlearning import Q_Learning_run
from genetik.ga import run_genetic_algorithm
from graph_and_metriks.graph_utils import create_graph_from_csv, create_random_graph

def get_graph():
    if os.path.exists(edge_path):
        return create_graph_from_csv(edge_path, demand_path, node_path)
    else:
        return create_random_graph(num_nodes=50)

def generate_scenarios(G, num_cases=20):
    nodes = list(G.nodes())
    scenarios = []
    
    for i in range(num_cases):
        s = random.choice(nodes)
        d = random.choice(nodes)
        while s == d:
            d = random.choice(nodes)
        
        if i < 18:
            bandwidth = random.randint(10, 150) 
        else:
            bandwidth = 999999

        mode = random.choice(['delay', 'reliability', 'balanced'])
        if mode == 'delay':
            weights = {"delay": 1.0, "reliability": 0.1, "resource": 0.1}
        elif mode == 'reliability':
            weights = {"delay": 0.1, "reliability": 1.0, "resource": 0.1}
        else:
            weights = {"delay": 0.5, "reliability": 0.5, "resource": 0.5}

        scenarios.append({
            "id": i + 1,
            "source": s,
            "destination": d,
            "bandwidth": bandwidth,
            "weights": weights,
            "mode": mode
        })
    return scenarios

def run_experiments():
    G = get_graph()
    scenarios = generate_scenarios(G, 20)
    
    results = []
    
    print(f"{'='*80}")
    print(f"{'TEST PROCESS':^80}")
    print(f"{'='*80}\n")

    for sc in scenarios:
        s, d, b, w = sc["source"], sc["destination"], sc["bandwidth"], sc["weights"]
        print(f"Scenario {sc['id']}: {s} -> {d} | Demand: {b} | Mode: {sc['mode']}")
        
        algos = ["Q-Learning", "Genetic"]
        
        for algo_name in algos:
            costs = []
            times = []
            success_count = 0
            
            for _ in range(5):
                start_time = time.time()
                path = None
                cost = float('inf')
                
                try:
                    if algo_name == "Q-Learning":
                        path, cost = Q_Learning_run(G, s, d, b, w)
                    elif algo_name == "Genetic":
                        path, cost = run_genetic_algorithm(G, s, d, b, w, pop_size=30, generations=20)
                except Exception:
                    path = None
                
                end_time = time.time()
                duration = end_time - start_time
                
                if path is not None and cost != float('inf'):
                    costs.append(cost)
                    times.append(duration)
                    success_count += 1
            
            if success_count > 0:
                avg_cost = np.mean(costs)
                std_dev = np.std(costs)
                best_res = np.min(costs)
                worst_res = np.max(costs)
                avg_time = np.mean(times)
                status = "SUCCESS"
            else:
                avg_cost = 0
                std_dev = 0
                best_res = 0
                worst_res = 0
                avg_time = np.mean(times) if times else 0
                status = "FAILURE"

            results.append({
                "Scenario ID": sc["id"],
                "Source": s,
                "Destination": d,
                "Demand": b,
                "Mode": sc["mode"],
                "Algorithm": algo_name,
                "Status": status,
                "Avg Time (s)": round(avg_time, 4),
                "Avg Cost": round(avg_cost, 2),
                "Std Dev": round(std_dev, 2),
                "Best": round(best_res, 2),
                "Worst": round(worst_res, 2)
            })
            
            print(f"   -> {algo_name:<12}: {status} (Avg Cost: {avg_cost:.2f}, Time: {avg_time:.4f}s)")

    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    print(f"\n{'='*80}")
    print("Completed. Saved to experiment_results.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_experiments()