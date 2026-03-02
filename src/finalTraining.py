from sklearn.model_selection import KFold
import pandas as pd
import torch
import matplotlib.pyplot as plt
# ---------------------------------------------------------
from graphCreator import GraphCreator
# ---------------------------------------------------------
from modelTrainer import EntrenadorGNN
from model import AdaptiveModel
# ---------------------------------------------------------
from predictionsVisualizer import plot_model_performance
# ---------------------------------------------------------

FINAL_MODEL_METRICS_PATH = './data/metrics/final_models/'
FINAL_MODELS_DIR = 'final_models'
TRAINING_MEDIA_PATH = './media/training/'

def main_tester(cfg, regen_graphs=False):
    # ---------------------------------------------------------
    # 1. CARGA DE DATOS
    # ---------------------------------------------------------
    graph_loader = GraphCreator()
    if regen_graphs:
        graph_loader.create_all_graphs()

    graph = None
    if cfg['model_type'] == 'STGNN' or cfg['model_type'] == 'LSTM':
        graph = graph_loader.load_graph(cat_opt='Temp', sim_profile='a&g', k_neighbors=5, dyn_graph=True)
    else:
        graph = graph_loader.load_graph(cat_opt='MP', sim_profile='a&g', k_neighbors=5, dyn_graph=False)

    num_students = graph.num_nodes
    print(f"Número de estudiantes (nodos): {num_students}")
    print(f"\tShape características estático: {graph.x.shape}")


    # --- CONFIGURACIÓN CV ---
    K_FOLDS = 5
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    trainer = EntrenadorGNN()

    all_results = []     # Resultados detallados (cada fold)
    summary_results = [] # Medias y desviación típica

    predictions_for_plot = {}  # Para gráficas: { 'Modelo': (y_true, y_pred) }
    
    print(f"🚀 Iniciando Cross-Validation ({K_FOLDS} Folds) para {cfg['model_type']}...\n")

    
    fold_metrics = []
    

    # Bucle de Folds
    for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(range(num_students))):
        # Convertir numpy indices a torch long
        train_idx = torch.tensor(train_idx_np, dtype=torch.long)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long)

        
        info_fold = {}
        metrics = {}
        metrics, y_true, y_pred, modelo, cfg, train_losses, val_losses = trainer.entrenar(
            graph, train_idx, val_idx, config=cfg)
       
        
        predictions_for_plot[cfg['model_type']] = (y_true, y_pred)

        info_fold['Model_Name'] = cfg['model_name']
        info_fold['Model_Type'] = cfg['model_type']
        if cfg['model_type'] == 'STGNN':
            info_fold['STGNN_Type'] = cfg['type_stgnn']
        info_fold['Model_Flexible'] = cfg['flexible']
        info_fold['Fold'] = fold + 1
        info_fold['Metrics'] = metrics
        info_fold['Config'] = cfg
        info_fold['Model'] = modelo
        info_fold['Train_Losses'] = train_losses
        info_fold['Val_Losses'] = val_losses
        fold_metrics.append(info_fold)
        

    df_fold = pd.DataFrame(fold_metrics)

    # Guardamos el modelo que mejor R2 haya tenido en los folds
    best_fold = df_fold.loc[df_fold['Metrics'].apply(lambda x: x['R2']).idxmax()]
    best_model = best_fold['Model']
    best_cfg = best_fold['Config']
    best_metrics = best_fold['Metrics']
    trainer.save_model(best_model, best_cfg, metrics=best_metrics, dir=FINAL_MODELS_DIR)

    # Gráfica de pérdidas (train vs val) del mejor fold
    train_vs_val_loss_plot(best_fold['Train_Losses'], best_fold['Val_Losses'], best_cfg['model_name'])

    summary = {
        'Nombre_Modelo':    cfg['model_name'],
        'Modelo':           cfg['model_type'],
        'Flexible':         cfg['flexible'],

        'MAE_Mean':         df_fold['Metrics'].apply(lambda x: x['MAE']).mean(),
        'MAE_Std':          df_fold['Metrics'].apply(lambda x: x['MAE']).std(),
        'Best_MAE':         best_metrics['MAE'],

        'RMSE_Mean':        df_fold['Metrics'].apply(lambda x: x['RMSE']).mean(),
        'RMSE_Std':         df_fold['Metrics'].apply(lambda x: x['RMSE']).std(),
        'Best_RMSE':        best_metrics['RMSE'],

        'MAPE_Mean':        df_fold['Metrics'].apply(lambda x: x['MAPE']).mean(),
        'MAPE_Std':         df_fold['Metrics'].apply(lambda x: x['MAPE']).std(),
        'Best_MAPE':        best_metrics['MAPE'],

        'R2_Mean':          df_fold['Metrics'].apply(lambda x: x['R2']).mean(),
        'R2_Std':           df_fold['Metrics'].apply(lambda x: x['R2']).std(),
        'Best_R2':          best_metrics['R2'],

        'Acc_Mean':         df_fold['Metrics'].apply(lambda x: x['Accuracy']).mean(),
        'Best_Acc':         best_metrics['Accuracy'],

        'F1_Mean':          df_fold['Metrics'].apply(lambda x: x['F1_Score']).mean(),
        'Best_F1':          best_metrics['F1_Score'],

        'Recall_Mean':      df_fold['Metrics'].apply(lambda x: x['Recall']).mean(),
        'Best_Recall':      best_metrics['Recall']
        
        
        
    }
    summary_results.append(summary)
    print(f"📊 Resumen {cfg['model_name']} ({'Flexible' if cfg['flexible'] else 'Rígido'}): MAE = {summary['MAE_Mean']:.3f} ± {summary['MAE_Std']:.3f}, MAPE = {summary['MAPE_Mean']:.3f} ± {summary['MAPE_Std']:.3f}\n")

    for r in fold_metrics:
        aux = {
            'Nombre_Modelo': r['Model_Name'],
            'Modelo': r['Model_Type'],
            'Flexible': r['Model_Flexible'],
            'Fold': r['Fold'],
            
            'R2': r['Metrics']['R2'],

            'MAE': r['Metrics']['MAE'],
            'RMSE': r['Metrics']['RMSE'],
            'MAPE': r['Metrics']['MAPE'],
            
            
            'Accuracy': r['Metrics']['Accuracy'],
            'F1_Score': r['Metrics']['F1_Score'],
            'Recall': r['Metrics']['Recall']
        }
        all_results.append(aux)
        

    print("\n🎨 Generando gráficas de rendimiento...")
    plot_model_performance(predictions_for_plot, tag=cfg['model_name'])
    # --- GUARDAR RESULTADOS ---
    df_detail = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_results)

    df_detail.to_csv(FINAL_MODEL_METRICS_PATH + f'{cfg['model_name']}_metrics.csv', index=False)
    df_summary.to_csv(FINAL_MODEL_METRICS_PATH + f'{cfg['model_name']}_summary.csv', index=False)

    print("✅ Proceso completado.")
    print("\nTabla Resumen:")
    print(df_summary.to_string(index=False))

def train_vs_val_loss_plot(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Train vs Validation Loss - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # Guardar la gráfica
    plt.savefig(TRAINING_MEDIA_PATH + f'{model_name}_train_val_loss.png')
    plt.show()


if __name__ == "__main__":
    cfg = {}
    cfg['model_type'] = input("Tipo de modelo (STGNN, LSTM, GCN, GAT, SAGE): ")
    if cfg['model_type'] == 'STGNN':
        cfg['type_stgnn'] = input("Tipo de STGNN (GCN, GAT, SAGE): ")
    cfg['model_name'] = cfg['model_type'] + '_final'
    if cfg['model_type'] == 'STGNN':
        cfg['model_name'] += f"_{cfg['type_stgnn']}"

    cfg['hidden_dim'] = int(input("Dimensión oculta: "))
    cfg['dropout'] = float(input("Dropout (0-1): "))
    cfg['lr'] = float(input("Learning Rate: "))

    flex = input("¿Entrenamiento flexible? (s/n): ").lower() == 's'
    cfg['flexible'] = flex

    patient = input("¿Modo paciente? (s/n): ").lower() == 's'
    if patient:
        cfg['epochs'] = 1000
        cfg['paciencia'] = 100
        cfg['max_restarts'] = 6

    regen_graphs = input("¿Regenerar grafos? (s/n): ").lower() == 's'

    main_tester(cfg, regen_graphs=regen_graphs)