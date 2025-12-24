from genetik.ga import run_genetic_algorithm


def Q_Learning_run(G, s, d, talep, weights):
    print("Q-learning Çalıştı")
    return run_genetic_algorithm(G, s, d, talep, weights)