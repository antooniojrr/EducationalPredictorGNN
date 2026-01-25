from model import AdaptiveModel, EntrenadorGNN
from graphCreator import GraphCreator
from sklearn.model_selection import KFold
import pandas as pd
import torch
from predictionsVisualizer import plot_model_performance

MODEL_METRICS_PATH = './data/metrics/model_metrics.csv'
MODEL_SUMMARY_PATH = './data/metrics/model_summary.csv'
def main():
   # ---------------------------------------------------------
    # 1. CARGA DE DATOS (Simulación - Usa tu DataLoader aquí)
    # ---------------------------------------------------------
    graph_loader = GraphCreator()
    #graph_loader.create_all_graphs()

    # Probamos con opción attendance & grades
    temp_graph = graph_loader.load_graph(cat_opt='Temp', sim_profile='a&g', k_neighbors=5, dyn_graph=True)
    static_graph = graph_loader.load_graph(cat_opt='MP', sim_profile='a&g', k_neighbors=5, dyn_graph=False)

    num_students = static_graph.num_nodes
    print(f"Número de estudiantes (nodos): {num_students}")
    print(f"\tShape características estático: {static_graph.x.shape}")
    print(f"\tShape características temporal: {temp_graph.x.shape}")


    # --- CONFIGURACIÓN CV ---
    K_FOLDS = 5
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    modelos = AdaptiveModel.TYPES
    trainer = EntrenadorGNN()

    all_results = []     # Resultados detallados (cada fold)
    summary_results = [] # Medias y desviación típica

    predictions_for_plot = {}  # Para gráficas: { 'Modelo': (y_true, y_pred) }
    print(f"🚀 Iniciando Cross-Validation ({K_FOLDS} Folds) para {len(modelos)} modelos...\n")

    for nombre_modelo in modelos:
        print(f"🔹 Evaluando: {nombre_modelo}")
        fold_metrics = []
        
        # Bucle de Folds
        for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(range(num_students))):
            # Convertir numpy indices a torch long
            train_idx = torch.tensor(train_idx_np, dtype=torch.long)
            val_idx = torch.tensor(val_idx_np, dtype=torch.long)

            
            metrics = {}
            if nombre_modelo == 'STGNN' or nombre_modelo == 'LSTM':
                metrics, y_true, y_pred = trainer.entrenar(
                    temp_graph, nombre_modelo, train_idx, val_idx
                )
            else:
                metrics, y_true, y_pred = trainer.entrenar(
                    static_graph, nombre_modelo, train_idx, val_idx
                )
            
            predictions_for_plot[nombre_modelo] = (y_true, y_pred)
            metrics['Modelo'] = nombre_modelo
            metrics['Fold'] = fold + 1
            fold_metrics.append(metrics)
            all_results.append(metrics)
            
        
        # Calcular Media y Std para este modelo
        df_fold = pd.DataFrame(fold_metrics)
        summary = {
            'Modelo': nombre_modelo,
            'MAE_Mean': df_fold['MAE'].mean(),
            'MAE_Std': df_fold['MAE'].std(),
            'R2_Mean': df_fold['R2'].mean(),
            'Acc_Mean': df_fold['Accuracy'].mean(),
            'F1_Mean': df_fold['F1_Score'].mean()
        }
        summary_results.append(summary)
        print(f"📊 Resumen {nombre_modelo}: MAE = {summary['MAE_Mean']:.3f} ± {summary['MAE_Std']:.3f}\n")

    print("\n🎨 Generando gráficas de rendimiento...")
    plot_model_performance(predictions_for_plot)
    # --- GUARDAR RESULTADOS ---
    df_detail = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_results)
    
    df_detail.to_csv(MODEL_METRICS_PATH, index=False)
    df_summary.to_csv(MODEL_SUMMARY_PATH, index=False)
    
    print("✅ Proceso completado.")
    print("\nTabla Resumen:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()