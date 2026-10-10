import pandas as pd

from src.kmeans import executar_kmeans
from src.metricas import avaliar_agrupamento


def avaliar_valores_k(dados, k_min=2, k_max=10):

    resultados = []

    for k in range(k_min, k_max + 1):

        # Executar o K-Means
        resultado = executar_kmeans(
            dados=dados,
            k=k,
            semente=42
        )

        # Calcular métricas de qualidade
        metricas = avaliar_agrupamento(
            dados,
            resultado["rotulos"]
        )

        # Armazenar resultados
        resultados.append({
            "k": k,
            "tempo": resultado["tempo_execucao"],
            "iteracoes": resultado["iteracoes"],
            "inercia": resultado["inercia"],
            "silhouette": metricas["silhouette"],
            "davies_bouldin": metricas["davies_bouldin"],
            "calinski_harabasz": metricas["calinski_harabasz"]
        })

    return pd.DataFrame(resultados)