import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_model_performance(predictions_dict, save_path='./media/predictions/'):
    """
    Genera gráficas comparativas Real vs Predicho para cada modelo.
    predictions_dict: { 'NombreModelo': (y_true, y_pred) }
    """
    os.makedirs(save_path, exist_ok=True)
    sns.set_style("whitegrid")
    
    # Colores consistentes
    colors = sns.color_palette("husl", len(predictions_dict))
    
    # ---------------------------------------------------------
    # GRÁFICA 1: SCATTER PLOTS (Predicción vs Realidad)
    # ---------------------------------------------------------
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
    
    if n_models == 1: 
        axes = [axes] # Asegurar iterable si solo hay 1 modelo
    
    for i, (model_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        ax = axes[i]
        
        # --- CORRECCIÓN AGRESIVA: RAVEL() ---
        # np.ravel() aplana cualquier dimensión extra (N,1,1) -> (N,)
        flat_true = np.ravel(y_true) * 10
        flat_pred = np.ravel(y_pred) * 10
        
        # Puntos
        sns.scatterplot(x=flat_true, y=flat_pred, ax=ax, color=colors[i], alpha=0.6, s=50)
        
        # Línea ideal (y=x)
        ax.plot([0, 10], [0, 10], 'r--', lw=2, label='Ideal')
        
        # Detalles
        ax.set_title(f"{model_name}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Nota Real (0-10)")
        if i == 0: ax.set_ylabel("Predicción (0-10)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(f"{save_path}scatter_comparativo.png", dpi=300)
    print(f"✅ Scatter plot guardado en {save_path}scatter_comparativo.png")
    plt.close()

    # ---------------------------------------------------------
    # GRÁFICA 2: DETALLE ORDENADO
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Usamos el primer modelo para referencia de orden
    first_key = list(predictions_dict.keys())[0]
    
    # --- CORRECCIÓN AQUÍ TAMBIÉN ---
    # Obtenemos y_true del primer modelo y lo aplanamos inmediatamente
    y_true_raw = predictions_dict[first_key][0]
    y_true_ref = np.ravel(y_true_raw)
    
    # Ordenamos índices de menor a mayor nota real
    sorted_indices = np.argsort(y_true_ref)
    
    # Pintamos la realidad (Línea negra de fondo)
    # Al usar y_true_ref (que ya es plano), sorted_indices funcionará bien
    plt.plot(y_true_ref[sorted_indices]*10, 'k-', lw=3, label='Realidad', alpha=0.3)
    
    for i, (model_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        # Aplanamos predicción actual
        y_pred_flat = np.ravel(y_pred)
        
        # Pintamos la predicción ordenada según la realidad
        plt.plot(y_pred_flat[sorted_indices]*10, 'o', markersize=4, label=f'Pred {model_name}', color=colors[i], alpha=0.7)
        
    plt.title("Comportamiento de los Modelos (Alumnos ordenados por nota real)", fontsize=16)
    plt.xlabel("Alumnos (ordenados de peor a mejor nota)", fontsize=12)
    plt.ylabel("Nota Final", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}lineplot_ordenado.png", dpi=300)
    print(f"✅ Line plot guardado en {save_path}lineplot_ordenado.png")
    plt.close()