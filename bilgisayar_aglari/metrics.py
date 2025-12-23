import math

def compute_metrics(G, path):
    """
    Verilen yol (path) için tüm QoS metriklerini hesaplar.
    """
    if path is None or len(path) < 2:
        return None

    # TOPLAM GECİKME HESAPLAMA
    link_delay_sum = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        link_delay_sum += G.edges[u, v]["delay"]

    proc_delay_sum = 0.0
    for node in path[1:-1]:
        proc_delay_sum += G.nodes[node]["processing_delay"]

    total_delay = link_delay_sum + proc_delay_sum

    # TOPLAM GÜVENİLİRLİK HESAPLAMA
    total_reliability = 1.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        total_reliability *= G.edges[u, v]["reliability"]
    for node in path:
        total_reliability *= G.nodes[node]["reliability"]

    #GÜVENİLİRLİK MALİYETİ
    reliability_cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        reliability_cost += -math.log(G.edges[u, v]["reliability"])
    for node in path:
        reliability_cost += -math.log(G.nodes[node]["reliability"])

    #KAYNAK KULLANIM MALİYETİ
    resource_cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        resource_cost += 1000.0 / G.edges[u, v]["bandwidth"]

    return {
        "total_delay": total_delay,
        "total_reliability": total_reliability,
        "reliability_cost": reliability_cost,
        "resource_cost": resource_cost,
    }

def total_cost(metrics, weights):
    """
    Ağırlıklı toplam yöntemini uygular.
    """
    return (
        weights["delay"] * metrics["total_delay"]
        + weights["reliability"] * metrics["reliability_cost"]
        + weights["resource"] * metrics["resource_cost"]
    )
