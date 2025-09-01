from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Função perform_grid_search (caso ainda não esteja definida)
def perform_grid_search(pipeline, param_grid, X, y, scoring='recall', cv=5, verbose=1):
    """
    Realiza o GridSearchCV para otimizar os hiperparâmetros de um pipeline.

    Args:
        pipeline (Pipeline): Pipeline com pré-processador e modelo.
        param_grid (dict): Dicionário com os hiperparâmetros a serem testados.
        X (DataFrame): Conjunto de dados (features).
        y (Series): Conjunto de dados (target).
        scoring (str): Métrica de avaliação (padrão: 'recall').
        cv (int): Número de dobras para validação cruzada (padrão: 5).
        verbose (int): Nível de detalhamento do progresso (padrão: 1).

    Returns:
        dict: Melhores parâmetros encontrados.
        float: Melhor pontuação obtida.
        Pipeline: Pipeline treinado com os melhores parâmetros.
    """
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    print("Melhores parâmetros:", grid_search.best_params_)
    print(f"Melhor {scoring}: {grid_search.best_score_:.4f}")
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_
