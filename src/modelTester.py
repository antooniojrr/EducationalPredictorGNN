from sklearn.model_selection import KFold
import pandas as pd
import torch
# ---------------------------------------------------------
from graphCreator import GraphCreator
# ---------------------------------------------------------
from modelTrainer import EntrenadorGNN
from model import AdaptiveModel
# ---------------------------------------------------------
from predictionsVisualizer import plot_model_performance
# ---------------------------------------------------------

MODEL_METRICS_PATH = './data/metrics/'


def main_tester(flexible_models=False, patient_mode=False, regen_graphs=False):
    # ---------------------------------------------------------
    # 1. CARGA DE DATOS
    # ---------------------------------------------------------
    graph_loader = GraphCreator()
    if regen_graphs:
        graph_loader.create_all_graphs()

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

    modelos = AdaptiveModel.TYPES if not flexible_models else ['STGNN', 'LSTM']
    trainer = EntrenadorGNN()

    all_results = []     # Resultados detallados (cada fold)
    summary_results = [] # Medias y desviación típica

    predictions_for_plot = {}  # Para gráficas: { 'Modelo': (y_true, y_pred) }
    
    id = input("Introduce indicador del entrenamiento: ")
    print(f"🚀 Iniciando Cross-Validation ({K_FOLDS} Folds) para {len(modelos)} modelos...\n")

    for tipo_modelo in modelos:
        print(f"🔹 Evaluando: {tipo_modelo}")
        fold_metrics = []
        cfg = {}
        cfg['model_name'] = f"{tipo_modelo}_{id}"
        cfg['model_type'] = tipo_modelo
        cfg['flexible'] = flexible_models
        if patient_mode:
            cfg['epochs'] = 1000
            cfg['paciencia'] = 100
            cfg['max_restarts'] = 6

        # Bucle de Folds
        for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(range(num_students))):
            # Convertir numpy indices a torch long
            train_idx = torch.tensor(train_idx_np, dtype=torch.long)
            val_idx = torch.tensor(val_idx_np, dtype=torch.long)

            
            info_fold = {}
            metrics = {}
            if tipo_modelo == 'STGNN' or tipo_modelo == 'LSTM':
                metrics, y_true, y_pred, modelo, cfg = trainer.entrenar(
                    temp_graph, train_idx, val_idx, config=cfg
                )
            else:
                metrics, y_true, y_pred, modelo, cfg = trainer.entrenar(
                    static_graph, train_idx, val_idx, config=cfg
                )
            
            predictions_for_plot[tipo_modelo] = (y_true, y_pred)

            info_fold['Model_Name'] = cfg['model_name']
            info_fold['Model_Type'] = tipo_modelo
            info_fold['Model_Flexible'] = cfg['flexible']
            info_fold['Fold'] = fold + 1
            info_fold['Metrics'] = metrics
            info_fold['Config'] = cfg
            info_fold['Model'] = modelo
            fold_metrics.append(info_fold)
            
    
        df_fold = pd.DataFrame(fold_metrics)

        # Guardamos el modelo que mejor R2 haya tenido en los folds
        best_fold = df_fold.loc[df_fold['Metrics'].apply(lambda x: x['R2']).idxmax()]
        best_model = best_fold['Model']
        best_cfg = best_fold['Config']
        best_metrics = best_fold['Metrics']
        trainer.save_model(best_model, best_cfg, metrics=best_metrics, dir=id)

        summary = {
            'Nombre_Modelo':    cfg['model_name'],
            'Modelo':           cfg['model_type'],
            'Flexible':         cfg['flexible'],
            'MAE_Mean':         df_fold['Metrics'].apply(lambda x: x['MAE']).mean(),
            'MAE_Std':          df_fold['Metrics'].apply(lambda x: x['MAE']).std(),
            'R2_Mean':          df_fold['Metrics'].apply(lambda x: x['R2']).mean(),
            'Acc_Mean':         df_fold['Metrics'].apply(lambda x: x['Accuracy']).mean(),
            'F1_Mean':          df_fold['Metrics'].apply(lambda x: x['F1_Score']).mean(),
            'Best_R2':          best_metrics['R2'],
            'Best_MAE':         best_metrics['MAE']
        }
        summary_results.append(summary)
        print(f"📊 Resumen {tipo_modelo} ({'Flexible' if cfg['flexible'] else 'Rígido'}): MAE = {summary['MAE_Mean']:.3f} ± {summary['MAE_Std']:.3f}\n")

        for r in fold_metrics:
            aux = {
                'Nombre_Modelo': r['Model_Name'],
                'Modelo': r['Model_Type'],
                'Flexible': r['Model_Flexible'],
                'Fold': r['Fold'],
                'MAE': r['Metrics']['MAE'],
                'R2': r['Metrics']['R2'],
                'Accuracy': r['Metrics']['Accuracy'],
                'F1_Score': r['Metrics']['F1_Score']
            }
            all_results.append(aux)
        

    print("\n🎨 Generando gráficas de rendimiento...")
    plot_model_performance(predictions_for_plot, tag=id)
    # --- GUARDAR RESULTADOS ---
    df_detail = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_results)

    df_detail.to_csv(MODEL_METRICS_PATH + f'{id}_metrics.csv', index=False)
    df_summary.to_csv(MODEL_METRICS_PATH + f'{id}_summary.csv', index=False)

    print("✅ Proceso completado.")
    print("\nTabla Resumen:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    flex = input("¿Modelos flexibles? (s/n): ").lower() == 's'
    patient = input("¿Modo paciente? (s/n): ").lower() == 's'
    regen_graphs = input("¿Regenerar grafos? (s/n): ").lower() == 's'
    main_tester(flex, patient, regen_graphs)