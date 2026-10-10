import time
import numpy as np
from sklearn.cluster import KMeans


def executar_kmeans(dados, k=3, semente=42):
    """
    Executa o algoritmo K-Means.

    Parâmetros:
        dados: DataFrame ou matriz NumPy
        k: número de clusters
        semente: semente aleatória

    Retorna:
        Dicionário com os resultados da execução.
    """

    # Converter os dados para NumPy
    X = np.asarray(dados, dtype=float)

    # Validar os dados
    if X.ndim != 2 or X.shape[1] != 24:
        raise ValueError(
            "A matriz deve possuir 24 atributos."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Os dados possuem valores ausentes ou inválidos."
        )

    if not 1 <= k <= len(X):
        raise ValueError(
            "Número de clusters inválido."
        )

    # Configurar o K-Means
    modelo = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=semente
    )

    # Medir o tempo de execução
    inicio = time.perf_counter()

    rotulos = modelo.fit_predict(X)

    fim = time.perf_counter()

    # Organizar os resultados
    resultados = {
        "rotulos": rotulos,
        "centroides": modelo.cluster_centers_,
        "tempo_execucao": fim - inicio,
        "iteracoes": modelo.n_iter_,
        "inercia": modelo.inertia_,
        "quantidade_clusters": len(
            np.unique(rotulos)
        )
    }

    return resultados