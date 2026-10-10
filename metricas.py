import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def avaliar_agrupamento(dados, rotulos):

    X = np.asarray(dados, dtype=float)

    quantidade_clusters = len(np.unique(rotulos))

    if not 2 <= quantidade_clusters < len(X):
        raise ValueError(
            "As métricas exigem entre 2 e n-1 clusters."
        )

    resultados = {
        "silhouette": silhouette_score(X, rotulos),
        "davies_bouldin": davies_bouldin_score(X, rotulos),
        "calinski_harabasz": calinski_harabasz_score(X, rotulos)
    }

    return resultados