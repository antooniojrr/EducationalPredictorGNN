"""
Módulo dedicado a la explicabilidad del modelo (XAI) para entender qué features y momentos temporales son más importantes para la predicción del rendimiento académico.
Contiene:
- feature_importance_analysis: Función que implementa la técnica de Permutation Feature Importance para evaluar la importancia de cada variable.
- week_importance_analysis: Función que evalúa la importancia de cada semana del curso permutando toda la información de esa semana.
- plot_feat_importance: Función para visualizar la importancia de las features.
- plot_week_importance: Función para visualizar la importancia temporal por semanas.
- Un bloque principal que carga un modelo entrenado, el grafo de datos y ejecuta los análisis de importancia, generando gráficos explicativos.
"""

import torch
from sklearn.metrics import r2_score
#----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
#----------------------------------------------------------
from modelTrainer import EntrenadorGNN
from graphCreator import GraphCreator

PATH_EXPLAINABILITY_OUTPUT = "./media/explainability/"
os.makedirs(PATH_EXPLAINABILITY_OUTPUT, exist_ok=True)

def feature_importance_analysis(model, data, feature_names=None):
    """
    Calcula la importancia de cada variable de entrada mediante
    permutation feature importance.

    El procedimiento consiste en:
        1. Calcular el rendimiento base del modelo (R²).
        2. Permutar los valores de una feature concreta entre nodos.
        3. Recalcular el rendimiento.
        4. Medir la caída en R² como indicador de importancia.

    Cuanto mayor sea la caída del rendimiento, mayor será la
    relevancia de la variable para el modelo.

    :param model: Modelo entrenado en modo evaluación.
    :param data: Objeto Data que contiene x (features) e y (targets).
                 Se asume formato temporal [N, T, F].
    :param feature_names: Lista opcional con nombres descriptivos
                          de las variables.
    :return: Diccionario {feature_name: importancia}.
    """
    model.eval()
    
    # 1. Cálculo del rendimiento base
    with torch.no_grad():
        original_pred = model(data)
        y_true = data.y.cpu().numpy().flatten()
        y_pred = original_pred.cpu().numpy().flatten()
        baseline_score = r2_score(y_true, y_pred)
    
    print(f"📊 R2 Original (Baseline): {baseline_score:.4f}")
    
    importances = {}
    
    # Se asume estructura temporal [N_alumnos, N_semanas, N_features]
    num_features = data.x.shape[2]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    # 2. Permutación independiente de cada variable
    for i in range(num_features):
        data_perturbed = data.clone()
        x_perturbed = data_perturbed.x.clone().cpu().numpy()
        
        feature_column = x_perturbed[:, :, i]
        np.random.shuffle(feature_column)
        x_perturbed[:, :, i] = feature_column
        
        data_perturbed.x = torch.tensor(x_perturbed).to(data.x.device)
        
        # 3. Evaluación tras perturbación
        with torch.no_grad():
            pred_perturbed = model(data_perturbed)
            y_pred_new = pred_perturbed.cpu().numpy().flatten()
            new_score = r2_score(y_true, y_pred_new)
        
        # 4. Cálculo de la importancia
        importance = baseline_score - new_score
        importances[feature_names[i]] = importance
        
        print(f"   -> Feature '{feature_names[i]}': R2 cae a {new_score:.4f} (Imp: {importance:.4f})")

    return importances


def week_importance_analysis(model, data):
    """
    Calcula la importancia temporal de cada semana en modelos
    con entrada secuencial.

    El método consiste en permutar toda la información correspondiente
    a una semana concreta entre nodos y medir la caída en el R².
    Esto permite estimar en qué momentos temporales el modelo
    basa con mayor peso su predicción final.

    Requiere que las entradas tengan dimensión 3:
        [N_nodos, N_semanas, N_features].

    :param model: Modelo entrenado.
    :param data: Objeto Data con información temporal.
    :return: Diccionario {"Semana i": importancia}.
    """
    model.eval()
    
    if data.x.dim() != 3:
        print("⚠️ ERROR: Para analizar importancia por semanas, necesitas datos TEMPORALES [N, Semanas, Features].")
        print(f"   Tu tensor actual es: {data.x.shape}")
        print("   Asegúrate de cargar el grafo con cat_opt='Temp'.")
        return {}

    # 1. Rendimiento base
    with torch.no_grad():
        original_pred = model(data)
        y_true = data.y.cpu().numpy().flatten()
        y_pred = original_pred.cpu().numpy().flatten()
        baseline_score = r2_score(y_true, y_pred)
    
    print(f"📊 R2 Original (Baseline): {baseline_score:.4f}")
    
    importances = {}
    num_weeks = data.x.shape[1]
    
    # 2. Permutación por instante temporal
    for w in range(num_weeks):
        data_perturbed = data.clone()
        x_perturbed = data_perturbed.x.clone()
        
        week_slice = x_perturbed[:, w, :]
        idx = torch.randperm(week_slice.size(0))
        x_perturbed[:, w, :] = week_slice[idx]
        
        data_perturbed.x = x_perturbed
        
        with torch.no_grad():
            pred_perturbed = model(data_perturbed)
            y_pred_new = pred_perturbed.cpu().numpy().flatten()
            new_score = r2_score(y_true, y_pred_new)
        
        importance = baseline_score - new_score
        importances[f"Semana {w+1}"] = importance
        
        print(f"   -> Semana {w+1}: R2 cae a {new_score:.4f} (Imp: {importance:.4f})")

    return importances


def plot_feat_importance(importances, save_path=PATH_EXPLAINABILITY_OUTPUT+"feature_importance.png"):
    """
    Genera y guarda un gráfico de barras con la importancia
    de cada variable de entrada.

    :param importances: Diccionario {feature: importancia}.
    :param save_path: Ruta donde se almacenará la figura.
    """
    df_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    
    plt.title("¿Qué features mira más el modelo?", fontsize=16, fontweight='bold')
    plt.xlabel("Caída del R² al permutar (Más es mejor)", fontsize=12)
    plt.ylabel("Variables", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Gráfico guardado en: {save_path}")
    plt.show()


def plot_week_importance(importances, save_path=PATH_EXPLAINABILITY_OUTPUT+"week_importance.png"):
    """
    Genera y guarda un gráfico de barras con la importancia
    temporal por semana.

    :param importances: Diccionario {"Semana i": importancia}.
    :param save_path: Ruta de almacenamiento de la figura.
    """
    weeks = list(importances.keys())
    vals = list(importances.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=weeks, y=vals, palette='rocket')
    plt.plot(weeks, vals, 'b-o', alpha=0.3, label='Tendencia')
    
    plt.title("Importancia Temporal: ¿Cuándo se decide la nota?", fontsize=16, fontweight='bold')
    plt.xlabel("Semanas del Curso", fontsize=12)
    plt.ylabel("Impacto en la Predicción (Drop in R²)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Gráfico guardado en: {save_path}")
    plt.show()


if __name__ == "__main__":
    """
    Script principal para ejecutar el análisis de explicabilidad.

    Flujo:
        1. Cargar grafo.
        2. Instanciar modelo entrenado.
        3. Ejecutar análisis de importancia por variables.
        4. Ejecutar análisis temporal (si procede).
        5. Generar visualizaciones.
    """

    MODELO_A_ANALIZAR = 'STGNN_26012026'
    CAT_OPT = 'Temp'
    
    print("Cargando grafo...")
    creator = GraphCreator()
    graph = creator.load_graph(cat_opt=CAT_OPT, sim_profile='a&g', k_neighbors=5, dyn_graph=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    nombres_features = creator.get_features_names()
    
    real_feats = 0
    if CAT_OPT == 'Temp':
        real_feats = graph.x.shape[2]
    else:
        real_feats = graph.x.shape[1]

    if len(nombres_features) != real_feats:
        print(f"⚠️ Aviso: Tienes {real_feats} features pero definiste {len(nombres_features)} nombres.")
        nombres_features = [f"Feat {i}" for i in range(real_feats)]
    
    print(f"Entrenando {MODELO_A_ANALIZAR} rápidamente para análisis...")
    trainer = EntrenadorGNN()
    
    model, _ = trainer.load_model(MODELO_A_ANALIZAR)
    
    print("Modelo listo. Iniciando XAI...")

    imps_feat = feature_importance_analysis(model, graph, feature_names=nombres_features)
    plot_feat_importance(imps_feat)
    
    if CAT_OPT == 'Temp':
        imps_weeks = week_importance_analysis(model, graph)
        plot_week_importance(imps_weeks)

    
        