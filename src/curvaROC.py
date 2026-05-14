"""Módulo para calcular la curva ROC y la AUC de un modelo de clasificación binaria."""

import matplotlib.pyplot as plt
# ---------------------------------------------------------
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import Data
# ---------------------------------------------------------
from model import AdaptiveModel
from modelTrainer import EntrenadorGNN
from dataLoader import DataLoader
from graphCreator import GraphCreator

def graficar_roc(test_true, test_pred, umbral_clase=7.0):
    """
    Calcula y grafica la Curva ROC para predicciones continuas.
    
    Args:
        test_true: Array con las notas reales (0-10).
        test_pred: Array con las notas predichas por el modelo (0-10).
        umbral_clase: Nota a partir de la cual se considera la clase positiva (Éxito = 1).
    """
    
    # Notas reales en 1 (Éxito) y 0 (Riesgo/Suspenso)
    y_true_binary = (test_true >= umbral_clase).astype(int)
    
    # Calcular FPR (Falsos Positivos) y TPR (Verdaderos Positivos) con la función roc_curve
    fpr, tpr, umbrales_roc = roc_curve(y_true_binary, test_pred)
    
    # Área Bajo la Curva (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Configurar y mostrar la gráfica
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='#00D2FF', lw=2.5, 
             label=f'Model STGNN-GAT (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='#2C3E50', lw=2, linestyle='--', 
             label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR = 1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR = Sensitivity)', fontsize=12)
    plt.title('ROC Curve - Academic Risk Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.savefig('./media/curva_roc_stgnn_english.pdf', format='pdf', bbox_inches='tight')
    
    plt.show()
    
    return roc_auc


def main():

    print("Cargando datos y modelo entrenado...")
    datos = GraphCreator().load_graph()

    modelo, _ = EntrenadorGNN().load_model(model_name="STGNN_final_GAT", flexible=True, dir="final_models")

    print("🔍 Analizando el grafo y calculando predicciones...")
    
    modelo.eval()
    pred = modelo(datos) 

    y_true = datos.y.numpy().flatten() * 10
    y_pred = pred.detach().numpy().flatten() * 10

    auc_value = graficar_roc(y_true, y_pred)
    print(f"AUC del modelo: {auc_value:.3f}")


if __name__ == "__main__":
    main()