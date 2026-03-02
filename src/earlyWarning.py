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

"""Este módulo implementa la clase EarlyWarningAnalyzer, que se encarga de realizar el estudio de evolución temporal para la alerta temprana."""
class EarlyWarningAnalyzer:

    def __init__(self, trainer, model):
        """Inicializa el analizador con el trainer y el modelo cargado.
        
        Args:
            trainer (EntrenadorGNN): Instancia del entrenador que se usó para entrenar el modelo (necesario para cargar el modelo correctamente).
            model: El modelo ya cargado que se va a analizar.
        """
        self.trainer = trainer
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {self.device}\n")
        self.model.eval()
        self.model.to(self.device)

    # ------------ FUNCIONES AUXILIARES PARA RECORTE DE DATOS ------------
    def _deterministic_slice(self, data, t):
        """
        Recorta los datos para simular que solo tenemos información hasta la semana 't'.
        Mantiene la estructura del grafo dinámico correspondiente.
        """
        # 1. Recortar Features [N, t, F]
        # Asumimos que data.x tiene forma [N, Semanas, Features]
        if data.x.shape[1] < t:
            return data # No se puede cortar más allá del total
            
        x_cut = data.x[:, :t, :]
        
        # 2. Recortar Grafos Dinámicos
        if hasattr(data, 'dynamic_edge_indices'):
            # Lista de adj matrices hasta t
            current_dyn_edges = data.dynamic_edge_indices[:t]
            # El grafo estático para GCN/GAT sería el de la semana t-1 (la última conocida)
            current_static_edge = data.dynamic_edge_indices[t-1]
        else:
            # Si el grafo es estático puro, no cambia
            current_dyn_edges = None
            current_static_edge = data.edge_index

        # Crear nuevo objeto Data
        # IMPORTANTE: data.y NO se toca. Queremos predecir la nota FINAL con datos PARCIALES.
        data_t = Data(x=x_cut, edge_index=current_static_edge, y=data.y)
        if current_dyn_edges:
            data_t.dynamic_edge_indices = current_dyn_edges
        
        return data_t.to(self.device)

    # ------------- FUNCIONES PRINCIPALES DE ESTUDIO Y PLOTEO -------------
    def run_study(self, data, test_idx=None, min_week=4, threshold=0.7):
        """
        Ejecuta el test progresivo.

        Args:
            data (Data): El objeto Data completo con todas las semanas.
            test_idx (list, optional): Índices de test para evaluar. Si es None, se evalúa sobre todo el dataset.
            min_week (int): Semana mínima para empezar el estudio.
            threshold (float): Umbral para binarizar la nota en la evaluación de recall (0-1).

        Returns:
            resultados (pd.DataFrame): DataFrame con las métricas calculadas para cada semana.
        """
        max_weeks = data.x.shape[1]
        results = []
        
        print(f"📉 Iniciando estudio de evolución temporal ({min_week} -> {max_weeks} semanas)...")
        y_true = None
        if test_idx is not None:
            y_true = data.y[test_idx].cpu().numpy().flatten()
        else:
            y_true = data.y.cpu().numpy().flatten()
        
        y_true_bin = (y_true >= threshold).astype(int)

        for t in range(min_week, max_weeks + 1):
            # 1. Preparar datos hasta semana t
            data_t = self._deterministic_slice(data, t)
            
            # 2. Inferencia
            with torch.no_grad():
                out = self.model(data_t)
                y_pred = None
                if test_idx:
                    y_pred = out[test_idx].cpu().numpy().flatten()
                else:
                    y_pred = out.cpu().numpy().flatten()
            
            # 3. Calcular Métricas
            # R2 (Regresión)
            # Desnormalizamos nota (0-1 -> 0-10) para R2 estándar
            r2 = r2_score(y_true * 10, y_pred * 10)
            
            # Recall (Clasificación)
            # Binarizar Predicción
            y_pred_bin = (y_pred >= threshold).astype(int)
            rec = recall_score(y_true_bin, y_pred_bin)
            
            results.append({
                'Semana': t,
                'R2': r2,
                'Recall': rec
            })
            print(f"\tSemana {t}: R2={r2:.4f}, Recall={rec:.4f}")

        return pd.DataFrame(results)

    def plot_results(self, df_results, name=None):
        """Genera las dos gráficas solicitadas.
        
        Args:
            df_results (pd.DataFrame): DataFrame con las columnas ['Semana', 'R2', 'Recall'].
            name (str, optional): Nombre del modelo para incluir en el título y nombre del archivo. Si es None, se omite.
        """
        sns.set(style="whitegrid")
        
        # --- GRÁFICA 1: Evolución del R2 ---
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

        # --- GRÁFICA 2: Evolución del Recall ---
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df_results, x='Semana', y='Recall', marker='s', linewidth=2.5, color='#c0392b')
        plt.title('Evolución de la Sensibilidad del Sistema (Recall)', fontsize=14, fontweight='bold')
        plt.xlabel('Semanas Transcurridas', fontsize=12)
        plt.ylabel('Recall (Clase Aprobado/Éxito)', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Zona de "éxito" visual
        plt.fill_between(df_results['Semana'], 0, df_results['Recall'], color='#c0392b', alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(f'./media/earlyWarning/evolucion_recall_{name if name else ""}.pdf', dpi=300)
        plt.show()

    def plot_all_results(self, df_results):
        """
        Genera gráficas comparativas con todas las líneas de los modelos juntos.
        Recibe un DataFrame que debe tener las columnas: ['Semana', 'R2', 'Recall', 'Modelo']

        Args:
            df_results (pd.DataFrame): DataFrame con las métricas de todos los modelos, incluyendo una columna 'Modelo' para diferenciarlos.
        """
        
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.2)
        
        # Paleta de colores distintiva
        palette = 'tab10' 

        # --- GRÁFICA 1: Comparativa R2 ---
        plt.figure(figsize=(12, 6))
        
        sns.lineplot(
            data=df_results, 
            x='Semana', 
            y='R2', 
            hue='Modelo',   # Diferencia por color
            style='Modelo', # Diferencia por tipo de línea/marcador
            markers=True, 
            dashes=False,   # Líneas sólidas para todos
            palette=palette,
            linewidth=2.5,
            markersize=9
        )
        
        plt.title('Evolución de la Precisión ($R^2$) - Comparativa de Modelos', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Semana del Curso', fontsize=14)
        plt.ylabel('Coeficiente $R^2$', fontsize=14)
        plt.ylim(0, 1.05)
        
        # Leyenda fuera para no tapar datos
        plt.legend(title='Arquitectura', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig('./media/comparativa_global_r2.png', dpi=300)
        print("✅ Gráfica comparativa R2 guardada.")
        plt.show()
        plt.close()

        # --- GRÁFICA 2: Comparativa Recall ---
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
        
        # Leyenda fuera
        plt.legend(title='Arquitectura', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig('./media/comparativa_global_recall.png', dpi=300)
        print("✅ Gráfica comparativa Recall guardada.")
        plt.show()
        plt.close()
        """Genera una gráfica comparativa de R2 y Recall para todos los modelos."""
        sns.set(style="whitegrid")
        
        # --- GRÁFICA COMPARATIVA: R2 ---
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_all_results, x='Semana', y='R2', hue='Modelo', marker='o', linewidth=2.5)
        plt.title('Comparativa de Capacidad Explicativa ($R^2$) entre Modelos', fontsize=14, fontweight='bold')
        plt.xlabel('Semanas Transcurridas', fontsize=12)
        plt.ylabel('Coeficiente $R^2$', fontsize=12)
        plt.ylim(0, 1.0)
        plt.legend(title='Modelo')
        plt.tight_layout()
        plt.savefig(f'./media/earlyWarning/comparativa_r2.pdf', dpi=300)
        plt.show()

        # --- GRÁFICA COMPARATIVA: Recall ---
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_all_results, x='Semana', y='Recall', hue='Modelo', marker='s', linewidth=2.5)
        plt.title('Comparativa de Sensibilidad (Recall) entre Modelos', fontsize=14, fontweight='bold')
        plt.xlabel('Semanas Transcurridas', fontsize=12)
        plt.ylabel('Recall (Clase Aprobado/Éxito)', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Zona de "éxito" visual
        for model in df_all_results['Modelo'].unique():
            subset = df_all_results[df_all_results['Modelo'] == model]
            plt.fill_between(subset['Semana'], 0, subset['Recall'], alpha=0.1)

        plt.legend(title='Modelo')
        plt.tight_layout()
        plt.savefig(f'./media/earlyWarning/comparativa_recall.pdf', dpi=300)
        plt.show()

# __________________________________________________________________________________________________________

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
        print(f"\n🔍 Analizando modelo: {model_name}...")
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
            print(f"❌ Error cargando {model_name}: {e}")

    # Concatenar resultados
    if results_list:
        df_all_results = pd.concat(results_list, ignore_index=True)
        
        analyzer.plot_all_results(df_all_results)
        
        print("\n📊 Tabla de resultados acumulados:")
        print(df_all_results.head())
    else:
        print("No se generaron resultados.")