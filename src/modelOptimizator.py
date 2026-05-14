"""
Módulo para la optimización de hiperparámetros mediante Grid Search con validación cruzada.
Este script permite evaluar sistemáticamente diferentes combinaciones de:
    - Dimensión oculta (hidden_dim)
    - Dropout
    - Learning rate
para distintos tipos de modelos (LSTM, GCN, GAT, SAGE, STGNN) y variantes flexibles.
El proceso incluye:
    1. Carga del grafo adecuado según el tipo de modelo.
    2. Configuración de validación cruzada K-Fold.
    3. Entrenamiento y evaluación para cada combinación de hiperparámetros.
    4. Resumen estadístico de los resultados por configuración.
    5. Identificación de la mejor configuración global.
    6. Exportación de resultados a un archivo CSV para análisis posterior.
El script es interactivo y permite seleccionar qué modelos optimizar, así como si se desea incluir variantes flexibles en la búsqueda.
"""

from sklearn.model_selection import KFold
import pandas as pd
import torch
import os
# ---------------------------------------------------------
from graphCreator import GraphCreator
# ---------------------------------------------------------
from modelTrainer import EntrenadorGNN
from model import AdaptiveModel

HIDDEN_DIM_OPTS= [16, 32, 64, 128]
DROPOUT_OPTS = [0.0, 0.2, 0.3, 0.5]
LR_OPTS = [0.1, 0.01, 0.005, 0.001]

RESULTS_CSV_PATH = './data/hyperparam_opt/'
os.makedirs(RESULTS_CSV_PATH, exist_ok=True)


def grid_adjustment(tipo_modelo, flexible_models = False, tipo_stgnn=None):
    """
    Realiza un proceso de optimización de hiperparámetros mediante
    Grid Search con validación cruzada K-Fold.

    Para cada combinación de:
        - Dimensión oculta (hidden_dim)
        - Dropout
        - Learning rate

    se entrena el modelo especificado y se evalúa su rendimiento
    promedio en validación cruzada. Los resultados agregados se
    almacenan en un archivo CSV.

    :param tipo_modelo: Tipo de modelo a optimizar
                        ('LSTM', 'GCN', 'GAT', 'SAGE', 'STGNN').
    :param flexible_models: Si True, activa configuración extendida
                            de entrenamiento (más épocas y reinicios).
    :param tipo_stgnn: Tipo de capa espacial en caso de optimizar STGNN
                       ('GCN', 'GAT', 'SAGE').
    """

    # ---------------------------------------------------------
    # 1. CARGA DEL GRAFO SEGÚN EL TIPO DE MODELO
    # ---------------------------------------------------------
    graph_loader = GraphCreator()

    graph = None
    try:
        if tipo_modelo in ['STGNN', 'LSTM']:
            graph = graph_loader.load_graph(
                cat_opt='Temp',
                sim_profile='a&g',
                k_neighbors=5,
                dyn_graph=True
            )
        else:
            graph = graph_loader.load_graph(
                cat_opt='MP',
                sim_profile='a&g',
                k_neighbors=5,
                dyn_graph=False
            )
    except Exception as e:
        print(f"Error al cargar el grafo para {tipo_modelo}: {e}")
        
    num_students = graph.num_nodes
    print(f"Número de estudiantes (nodos): {num_students}")
    print(f"\tShape características: {graph.x.shape}")

    # ---------------------------------------------------------
    # 2. CONFIGURACIÓN DE VALIDACIÓN CRUZADA
    # ---------------------------------------------------------
    K_FOLDS = 5
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    trainer = EntrenadorGNN()

    # ---------------------------------------------------------
    # 3. GRID SEARCH SOBRE HIPERPARÁMETROS
    # ---------------------------------------------------------
    results_grid = []
    best_config = None
    best_mean_r2 = -float('inf')

    print(f"🚀 Iniciando Grid Search para {tipo_modelo}{' (' + tipo_stgnn + ')' if tipo_stgnn else ''}...\n")

    for hidden_dim in HIDDEN_DIM_OPTS:
        for dropout in DROPOUT_OPTS:
            for lr in LR_OPTS:

                print(f"Evaluando configuración: Hidden Dim={hidden_dim}, Dropout={dropout}, LR={lr}")

                cfg = {}
                cfg['model_name'] = f"{tipo_modelo}{('_' + tipo_stgnn) if tipo_stgnn and tipo_modelo == 'STGNN' else ''}_{hidden_dim}_{dropout}_{lr}_{'flex' if flexible_models else ''}"
                cfg['model_type'] = tipo_modelo
                cfg['flexible'] = flexible_models
                cfg['hidden_dim'] = hidden_dim
                cfg['dropout'] = dropout
                cfg['lr'] = lr

                if tipo_modelo == 'STGNN' and tipo_stgnn:
                    cfg['stgnn_type'] = tipo_stgnn

                if flexible_models:
                    cfg['epochs'] = 1000
                    cfg['paciencia'] = 100
                    cfg['max_restarts'] = 6

                fold_metrics = []

                # ---------------------------------------------------------
                # 4. VALIDACIÓN CRUZADA
                # ---------------------------------------------------------
                for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(range(num_students))):

                    train_idx = torch.tensor(train_idx_np, dtype=torch.long)
                    val_idx = torch.tensor(val_idx_np, dtype=torch.long)

                    metrics, _, _, modelo, cfg = trainer.entrenar(
                        graph, train_idx, val_idx, config=cfg, verbose=False
                    )

                    info_fold = {}
                    info_fold['Model_Name'] = cfg['model_name']
                    info_fold['Model_Type'] = tipo_modelo
                    info_fold['Model_Flexible'] = cfg['flexible']
                    info_fold['Fold'] = fold + 1
                    info_fold['Metrics'] = metrics
                    info_fold['Config'] = cfg
                    info_fold['Model'] = modelo

                    fold_metrics.append(info_fold)
                
                df_fold = pd.DataFrame(fold_metrics)

                # Selección del mejor fold según R²
                best_fold = df_fold.loc[
                    df_fold['Metrics'].apply(lambda x: x['R2']).idxmax()
                ]
                best_metrics = best_fold['Metrics']

                # ---------------------------------------------------------
                # 5. RESUMEN ESTADÍSTICO DE LA CONFIGURACIÓN
                # ---------------------------------------------------------
                summary = {
                    'Nombre_Modelo':    cfg['model_name'],
                    'Modelo':           cfg['model_type'],
                    'Flexible':         cfg['flexible'],
                    'MAE_Mean':         df_fold['Metrics'].apply(lambda x: x['MAE']).mean(),
                    'MAE_Std':          df_fold['Metrics'].apply(lambda x: x['MAE']).std(),
                    'MAPE_Mean':        df_fold['Metrics'].apply(lambda x: x['MAPE']).mean(),
                    'MAPE_Std':         df_fold['Metrics'].apply(lambda x: x['MAPE']).std(),
                    'R2_Mean':          df_fold['Metrics'].apply(lambda x: x['R2']).mean(),
                    'Acc_Mean':         df_fold['Metrics'].apply(lambda x: x['Accuracy']).mean(),
                    'F1_Mean':          df_fold['Metrics'].apply(lambda x: x['F1_Score']).mean(),
                    'Best_R2':          best_metrics['R2'],
                    'Best_MAE':         best_metrics['MAE'],
                    'Best_MAPE':        best_metrics['MAPE'],
                    'Hidden_Dim':       hidden_dim,
                    'Dropout':          dropout,
                    'LR':               lr
                }

                # Actualización de la mejor configuración global
                if summary['R2_Mean'] > best_mean_r2:
                    best_mean_r2 = summary['R2_Mean']
                    best_config = cfg

                results_grid.append(summary)

                print(
                    f"✅ Configuración evaluada: Hidden Dim={hidden_dim}, "
                    f"Dropout={dropout}, LR={lr} "
                    f"{'(FLEXIBLE)' if flexible_models else ''}"
                    f"-> R2 Promedio = {summary['R2_Mean']:.4f}\n"
                )

    # ---------------------------------------------------------
    # 6. RESULTADO FINAL
    # ---------------------------------------------------------
    print(
        f"🎯 Mejor configuración encontrada: "
        f"Hidden Dim={best_config['hidden_dim']}, "
        f"Dropout={best_config['dropout']}, "
        f"LR={best_config['lr']} "
        f"con R2 Promedio = {best_mean_r2:.4f}\n"
    )

    # ---------------------------------------------------------
    # 7. EXPORTACIÓN DE RESULTADOS
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results_grid)

    path = os.path.join(
        RESULTS_CSV_PATH,
        f"grid_search_results_{tipo_modelo}"
        f"{'_' + tipo_stgnn if tipo_stgnn and tipo_modelo == 'STGNN' else ''}"
    )

    if flexible_models:
        path += "_flexible"

    path += ".csv"

    df_results.to_csv(path, index=False)

    print(f"📁 Resultados de Grid Search guardados en '{path}'\n")

if __name__ == "__main__":
    """
    Punto de entrada interactivo para la optimización
    de hiperparámetros.

    Permite:
        - Optimizar un modelo concreto.
        - Optimizar todos los modelos.
        - Optimizar únicamente modelos temporales.
        - Optimizar variantes específicas de STGNN.
    """

    modelos = AdaptiveModel.TYPES

    tipo_modelo = input(
        f"Introduzca el modelo a optimizar({modelos}), "
        f"'all' para optimizar todos o "
        f"'temp' para optimizar solo temporal: "
    )

    flex = input("¿Desea optimizar modelos flexibles? (s/n): ").lower() == 's'

    if tipo_modelo == 'all':
        for m in modelos:
            grid_adjustment(m, flexible_models=flex)

    elif tipo_modelo == 'temp':
        grid_adjustment('LSTM', flexible_models=flex)
        grid_adjustment('STGNN', flexible_models=flex)

    elif tipo_modelo == 'stgnn_opt':
        for stgnn_type in ['GCN', 'SAGE']:
            grid_adjustment('STGNN', flexible_models=flex, tipo_stgnn=stgnn_type)

    elif tipo_modelo in modelos:
        grid_adjustment(tipo_modelo, flexible_models=flex)

    else:
        print(f"Modelo no reconocido. Por favor, elija entre {modelos} o 'all'.")