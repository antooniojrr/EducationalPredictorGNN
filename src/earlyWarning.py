import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# --------------------------------
import torch
from sklearn.metrics import r2_score, recall_score
from torch_geometric.data import Data
# --------------------------------
from modelTrainer import EntrenadorGNN
from graphCreator import GraphCreator


class EarlyWarningAnalyzer:
    """
    Clase responsable del análisis de alerta temprana mediante un estudio
    de evolución temporal.

    Permite evaluar cómo varía el rendimiento predictivo de un modelo
    conforme se dispone progresivamente de más semanas de información,
    simulando un escenario real de predicción anticipada.
    """

    def __init__(self, trainer, model):
        """
        Inicializa el analizador con un modelo previamente entrenado.

        Args:
            trainer (EntrenadorGNN): Instancia del entrenador utilizada
                                     para gestionar el modelo.
            model (torch.nn.Module): Modelo ya cargado que será evaluado.
        """
        self.trainer = trainer
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {self.device}\n")
        self.model.eval()
        self.model.to(self.device)


    def _deterministic_slice(self, data, t):
        """
        Genera una versión recortada del grafo de datos que contiene
        únicamente la información disponible hasta la semana t.

        Este procedimiento simula un escenario de predicción temprana,
        manteniendo la coherencia estructural del grafo dinámico.

        Args:
            data (Data): Objeto completo con todas las semanas disponibles.
            t (int): Número de semanas consideradas.

        Returns:
            Data: Nuevo objeto Data limitado temporalmente hasta la semana t.
        """
        if data.x.shape[1] < t:
            return data
            
        x_cut = data.x[:, :t, :]
        
        if hasattr(data, 'dynamic_edge_indices'):
            current_dyn_edges = data.dynamic_edge_indices[:t]
            current_static_edge = data.dynamic_edge_indices[t-1]
        else:
            current_dyn_edges = None
            current_static_edge = data.edge_index

        data_t = Data(x=x_cut, edge_index=current_static_edge, y=data.y)
        if current_dyn_edges:
            data_t.dynamic_edge_indices = current_dyn_edges
        
        return data_t.to(self.device)


    def run_study(self, data, test_idx=None, min_week=4, threshold=0.7):
        """
        Ejecuta el estudio progresivo de alerta temprana.

        Para cada semana desde `min_week` hasta el total disponible,
        se evalúa el modelo utilizando únicamente la información acumulada
        hasta ese instante.

        Métricas calculadas:
            - R² (regresión sobre nota final desnormalizada).
            - Recall (clasificación binaria según umbral).

        Args:
            data (Data): Grafo completo con todas las semanas.
            test_idx (list | None): Índices del conjunto de test. Si es None,
                                    se evalúa sobre todos los nodos.
            min_week (int): Semana mínima desde la que comienza el análisis.
            threshold (float): Umbral en [0,1] para binarizar la nota final.

        Returns:
            pd.DataFrame: Tabla con columnas ['Semana', 'R2', 'Recall'].
        """
        max_weeks = data.x.shape[1]
        results = []
        
        print(f"Iniciando el estudio de evolución temporal desde semana {min_week} hasta {max_weeks}...")
        
        if test_idx is not None:
            y_true = data.y[test_idx].cpu().numpy().flatten()
        else:
            y_true = data.y.cpu().numpy().flatten()
        
        y_true_bin = (y_true >= threshold).astype(int)

        for t in range(min_week, max_weeks + 1):
            data_t = self._deterministic_slice(data, t)
            
            with torch.no_grad():
                out = self.model(data_t)
                if test_idx:
                    y_pred = out[test_idx].cpu().numpy().flatten()
                else:
                    y_pred = out.cpu().numpy().flatten()
            
            r2 = r2_score(y_true * 10, y_pred * 10)
            y_pred_bin = (y_pred >= threshold).astype(int)
            rec = recall_score(y_true_bin, y_pred_bin)
            
            results.append({
                'Semana': t,
                'R2': r2,
                'Recall': rec
            })

            print(f"Semana {t}: R²={r2:.4f}, Recall={rec:.4f}")

        return pd.DataFrame(results)


    def plot_results(self, df_results, name=None):
        """
        Genera gráficas individuales de evolución temporal para un modelo.

        Se producen dos figuras:
            1. Evolución del coeficiente R².
            2. Evolución del Recall.

        Args:
            df_results (pd.DataFrame): DataFrame con columnas
                                       ['Semana', 'R2', 'Recall'].
            name (str | None): Identificador del modelo para incluir
                               en el nombre del archivo de salida.
        """
        sns.set(style="whitegrid")
        
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_results, x='Semana', y='R2', marker='o', linewidth=2.5, color='#2980b9')
        plt.title('Incremento de la Capacidad Explicativa ($R^2$)', fontsize=14, fontweight='bold')
        plt.xlabel('Semanas Transcurridas', fontsize=12)
        plt.ylabel('Coeficiente $R^2$', fontsize=12)
        plt.ylim(0, 1.0)
        plt.axhline(y=df_results['R2'].max(), color='gray', linestyle='--', alpha=0.5, label='Máximo alcanzado')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./media/earlyWarning/evolucion_r2_{name if name else ""}.pdf', dpi=300)
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_results, x='Semana', y='Recall', marker='s', linewidth=2.5, color='#c0392b')
        plt.title('Evolución de la Sensibilidad del Sistema (Recall)', fontsize=14, fontweight='bold')
        plt.xlabel('Semanas Transcurridas', fontsize=12)
        plt.ylabel('Recall (Clase Aprobado/Éxito)', fontsize=12)
        plt.ylim(0, 1.0)
        plt.fill_between(df_results['Semana'], 0, df_results['Recall'], color='#c0392b', alpha=0.1)
        plt.tight_layout()
        plt.savefig(f'./media/earlyWarning/evolucion_recall_{name if name else ""}.pdf', dpi=300)
        plt.show()


    def plot_all_results(self, df_results):
        """
        Genera gráficas comparativas de múltiples modelos.

        El DataFrame debe contener las columnas:
            ['Semana', 'R2', 'Recall', 'Modelo'].

        Produce:
            - Comparativa global de R².
            - Comparativa global de Recall.

        Args:
            df_results (pd.DataFrame): Resultados agregados de todos los modelos.
        """
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        palette = 'tab10' 

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_results, 
            x='Semana', 
            y='R2', 
            hue='Modelo',
            style='Modelo',
            markers=True, 
            dashes=False,
            palette=palette,
            linewidth=2.5,
            markersize=9
        )
        
        plt.title('Evolución de la Precisión ($R^2$) - Comparativa de Modelos', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Semana del Curso', fontsize=14)
        plt.ylabel('Coeficiente $R^2$', fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(title='Arquitectura', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig('./media/comparativa_global_r2.png', dpi=300)
        print("Gráfica comparativa de R² guardada en disco.")
        plt.show()
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_results, 
            x='Semana', 
            y='Recall', 
            hue='Modelo',
            style='Modelo',
            markers=True, 
            dashes=False,
            palette=palette,
            linewidth=2.5,
            markersize=9
        )
        
        plt.title('Evolución de la Alerta Temprana (Recall) - Comparativa', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Semana del Curso', fontsize=14)
        plt.ylabel('Recall (Sensibilidad)', fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(title='Arquitectura', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig('./media/comparativa_global_recall.png', dpi=300)
        print("Gráfica comparativa de Recall guardada en disco.")
        plt.show()
        plt.close()

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":

    creator = GraphCreator()
    graph = creator.load_graph(cat_opt='Temp', sim_profile='a&g', k_neighbors=5, dyn_graph=True)
    
    model_names = [
        'LSTM_final_flexible', 
        'STGNN_final_GCN_flexible', 
        'STGNN_final_GAT_flexible', 
        'STGNN_final_SAGE_flexible'
    ]
    
    trainer = EntrenadorGNN()
    results_list = [] # Lista para acumular DataFrames
    
    
    for model_name in model_names:
        print(f"\nAnalizando modelo: {model_name}...")
        try:
            # Cargar modelo específico
            # Asumimos que load_model devuelve (modelo, config)
            model, cfg = trainer.load_model(model_name, dir='final_models')
            
            # Instanciar Analyzer
            analyzer = EarlyWarningAnalyzer(trainer, model)
            
            # Ejecutar estudio
            res_df = analyzer.run_study(graph, min_week=4)
            
            # Añadir etiqueta del modelo
            sp = model_name.split('_')
            name = sp[0]
            if name == "STGNN":
                name += f"-{sp[2]}"
            
            res_df['Modelo'] = name
            results_list.append(res_df)
            
        except Exception as e:
            print(f"Error al cargar el modelo {model_name}: {e}")

    # Concatenar resultados
    if results_list:
        df_all_results = pd.concat(results_list, ignore_index=True)
        
        analyzer.plot_all_results(df_all_results)
        
        print("\nTabla de resultados acumulados:")
        print(df_all_results.head())
    else:
        print("No se generaron resultados.")