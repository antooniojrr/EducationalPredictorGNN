import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd
from copy import deepcopy
import os

# Importa tus clases
from modelTrainer import EntrenadorGNN
from graphCreator import GraphCreator

PATH_EXPLAINABILITY_OUTPUT = "./media/explainability/"
os.makedirs(PATH_EXPLAINABILITY_OUTPUT, exist_ok=True)

def feature_importance_analysis(model, data, feature_names=None):
    """
    Calcula la importancia de cada feature permutando sus valores y midiendo la caída del rendimiento.
    """
    model.eval()
    
    # 1. Obtener línea base (Rendimiento original)
    with torch.no_grad():
        original_pred = model(data)
        y_true = data.y.cpu().numpy().flatten()
        y_pred = original_pred.cpu().numpy().flatten()
        baseline_score = r2_score(y_true, y_pred)
    
    print(f"📊 R2 Original (Baseline): {baseline_score:.4f}")
    
    importances = {}
    
    # Detectar dimensiones: [N_Alumnos, N_Semanas, N_Features]
    # Asumimos que data.x es [N, W, F]
    num_features = data.x.shape[2]
    
    # Si no hay nombres, generamos genéricos
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    # 2. Bucle de Permutación
    for i in range(num_features):
        # Hacemos una copia profunda de los datos para no romper el original
        data_perturbed = data.clone()
        x_perturbed = data_perturbed.x.clone().cpu().numpy()
        
        # --- LA MAGIA: Shuffle de la feature 'i' a través de todos los alumnos ---
        # Mantenemos la estructura temporal, pero intercambiamos alumnos para esta feature específica
        # x_perturbed[:, :, i] -> Shape [70, 12]
        
        feature_column = x_perturbed[:, :, i] 
        np.random.shuffle(feature_column) # Mezclamos filas (alumnos)
        x_perturbed[:, :, i] = feature_column
        
        # Devolvemos al tensor
        data_perturbed.x = torch.tensor(x_perturbed).to(data.x.device)
        
        # 3. Evaluar modelo roto
        with torch.no_grad():
            pred_perturbed = model(data_perturbed)
            y_pred_new = pred_perturbed.cpu().numpy().flatten()
            new_score = r2_score(y_true, y_pred_new)
        
        # 4. Importancia = Cuánto ha bajado el R2
        # (Baseline - New). Si baja mucho, la diferencia es grande (Importante).
        # Si sube (raro), es 0.
        importance = baseline_score - new_score
        importances[feature_names[i]] = importance
        
        print(f"   -> Feature '{feature_names[i]}': R2 cae a {new_score:.4f} (Imp: {importance:.4f})")

    return importances

def week_importance_analysis(model, data):
    """
    Calcula la importancia de cada SEMANA permutando toda la información de esa semana
    (Features, Asistencia, etc.) entre los alumnos y midiendo la caída del rendimiento.
    """
    model.eval()
    
    # Validar que los datos sean temporales (3D)
    if data.x.dim() != 3:
        print("⚠️ ERROR: Para analizar importancia por semanas, necesitas datos TEMPORALES [N, Semanas, Features].")
        print(f"   Tu tensor actual es: {data.x.shape}")
        print("   Asegúrate de cargar el grafo con cat_opt='Temp'.")
        return {}

    # 1. Obtener línea base
    with torch.no_grad():
        original_pred = model(data)
        y_true = data.y.cpu().numpy().flatten()
        y_pred = original_pred.cpu().numpy().flatten()
        baseline_score = r2_score(y_true, y_pred)
    
    print(f"📊 R2 Original (Baseline): {baseline_score:.4f}")
    
    importances = {}
    num_weeks = data.x.shape[1] # [N, Weeks, Features]
    
    # 2. Bucle por Semana
    for w in range(num_weeks):
        # Copia profunda
        data_perturbed = data.clone()
        x_perturbed = data_perturbed.x.clone() # Tensor torch
        
        # --- LA MAGIA: Shuffle de la SEMANA 'w' ---
        # Cogemos la "rebanada" de la semana w para TODOS los alumnos
        week_slice = x_perturbed[:, w, :] # Shape [N, Features]
        
        # Generamos índices aleatorios para mezclar los alumnos SOLO en esta semana
        idx = torch.randperm(week_slice.size(0))
        
        # Asignamos la semana mezclada (Rompe la relación causal para esta semana)
        x_perturbed[:, w, :] = week_slice[idx]
        
        data_perturbed.x = x_perturbed
        
        # 3. Evaluar
        with torch.no_grad():
            pred_perturbed = model(data_perturbed)
            y_pred_new = pred_perturbed.cpu().numpy().flatten()
            new_score = r2_score(y_true, y_pred_new)
        
        # 4. Importancia
        importance = baseline_score - new_score
        importances[f"Semana {w+1}"] = importance
        
        print(f"   -> Semana {w+1}: R2 cae a {new_score:.4f} (Imp: {importance:.4f})")

    return importances

def plot_feat_importance(importances, save_path=PATH_EXPLAINABILITY_OUTPUT+"feature_importance.png"):
    # Convertir a DataFrame para Seaborn
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
    weeks = list(importances.keys())
    vals = list(importances.values())
    
    plt.figure(figsize=(10, 6))
    
    # Gráfico de líneas o barras. Para tiempo, a veces línea mola más, pero barras es más claro para "importancia".
    sns.barplot(x=weeks, y=vals, palette='rocket')
    
    # Línea de tendencia (opcional)
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
    # --- CONFIGURACIÓN ---
    MODELO_A_ANALIZAR = 'STGNN_26012026' # O el que te haya dado mejor resultado
    CAT_OPT = 'Temp'  # 'MP' o 'Temp' según el grafo que usaste
    
    # 1. Cargar Datos y Grafo
    print("Cargando grafo...")
    creator = GraphCreator()
    # Usa la misma configuración que en tu main.py
    graph = creator.load_graph(cat_opt=CAT_OPT, sim_profile='a&g', k_neighbors=5, dyn_graph=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    # 2. Definir Nombres de Features (IMPORTANTE: Ajusta esto a tu orden real)
    # Según tu dataLoader, concatenas: [Asistencia (1), Notas (1), Encuestas (6?)]
    # Revisa dataLoader.py -> X_base = torch.cat([att, grades, surv], dim=2)
    nombres_features = creator.get_features_names()
    
    # Si la longitud no coincide, cortamos o rellenamos para que no falle
    real_feats = 0
    if CAT_OPT == 'Temp':
        real_feats = graph.x.shape[2]
    else:
        real_feats = graph.x.shape[1]

    if len(nombres_features) != real_feats:
        print(f"⚠️ Aviso: Tienes {real_feats} features pero definiste {len(nombres_features)} nombres.")
        nombres_features = [f"Feat {i}" for i in range(real_feats)]
    
    # 3. Cargar Modelo Entrenado
    # Aquí necesitamos instanciar el modelo y (idealmente) cargar pesos guardados.
    # Como tu código actual no guarda el .pth en disco permanentemente (solo en memoria durante el loop),
    # vamos a hacer un entrenamiento rápido de 1 fold para tener el modelo "caliente".
    print(f"Entrenando {MODELO_A_ANALIZAR} rápidamente para análisis...")
    trainer = EntrenadorGNN()
    
    model, _ = trainer.load_model(MODELO_A_ANALIZAR)
    
    print("Modelo listo. Iniciando XAI...")

    # 4. Análisis
    imps_feat = feature_importance_analysis(model, graph, feature_names=nombres_features)
    plot_feat_importance(imps_feat)
    
    if CAT_OPT == 'Temp':
        imps_weeks = week_importance_analysis(model, graph)
        plot_week_importance(imps_weeks)

    
        