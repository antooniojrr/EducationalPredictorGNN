import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

GRAPH_DIR = './data/processed/graphs/'
METRICS_MEDIA_DIR = './media/graphMetrics/'
os.makedirs(METRICS_MEDIA_DIR, exist_ok=True)

class GraphTester:        

    def load_graph(self):
        """Carga el grafo desde el archivo especificado."""
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"El archivo {self.graph_path} no existe.")
        
        g = torch.load(self.graph_path, weights_only=False)
        #print(f"Grafo cargado con {g.num_nodes} nodos y {g.num_edges} aristas.")
        return g

    def test_graph(self, graph_path: str = None):
        """Realiza pruebas básicas en el grafo cargado y devuelve las métricas obtenidas."""
        while graph_path is None:
            graph_path = self._select_graph()
            
        self.graph_path = graph_path
        self.graph: Data = self.load_graph()
        
        
        metrics = {}
        print("------------------TESTEANDO GRAFO------------------")
        # NODOS AISLADOS
        connected_nodes = torch.unique(self.graph.edge_index)
        num_connected = connected_nodes.size(0)

        num_isolated = self.graph.num_nodes - num_connected
        metrics['num_isolated_nodes'] = num_isolated

        if num_isolated > 0:
            print(f"TEST DE AISLAMIENTO: Advertencia, Hay {num_isolated} ({num_isolated/self.graph.num_nodes*100:.2f}%) nodos aislados en el grafo.")
        else:
            print("TEST DE AISLAMIENTO: No hay nodos aislados en el grafo.")

        # Comprobar la varianza de las tags de los nodos adyacentes
        var, std = self.calculate_neighbor_variance()
        metrics['neighbor_variance'] = var
        metrics['neighbor_std_dev'] = std

        print("TEST DE ADYACENCIA: Varianza de etiquetas entre vecinos calculada.")
        print(f"\t✅ Varianza Media de Vecinos: {(var*100.0):.4f}")
        print(f"\t📉 Desviación Típica Promedio: {(std*10.0):.4f} puntos")

        if std < 0.15:
            print("\t🚀 CONCLUSIÓN: El grafo tiene ALTA coherencia (Homofilia fuerte).")
        elif std < 0.25:
            print("\t⚠️ CONCLUSIÓN: El grafo tiene coherencia MEDIA.")
        else:
            print("\t❌ CONCLUSIÓN: El grafo es RUIDOSO (Vecinos con notas muy dispares).")
        
        # Comprobar outliers
        outliers = self.get_graph_outliers(threshold=0.3)
        metrics['num_outliers'] = len(outliers)
        print(f"TEST DE OUTLIERS: Se han detectado {len(outliers)} ({len(outliers)/self.graph.num_nodes*100:.2f}%) alumnos con notas considerablemente distintas a sus vecinos (>3 puntos).")

        # Comprobar conexiones en cada snapshot (si es dinámico)
        if hasattr(self.graph, 'dynamic_edge_indices'):
            print("TEST DE DINAMICIDAD: Comprobando conexiones en cada snapshot...")
            acc_var = 0.0
            acc_std = 0.0
            max_var = -float('inf')
            min_var = float('inf')
            for t, edge_index_t in enumerate(self.graph.dynamic_edge_indices):
                print(f"\tSEMANA {t}:")
                connected_nodes_t = torch.unique(edge_index_t)
                num_connected_t = connected_nodes_t.size(0)
                num_isolated_t = self.graph.num_nodes - num_connected_t

                var, std = self.calculate_neighbor_variance(week=t)
                if var > max_var:
                    max_var = var
                if var < min_var:
                    min_var = var
                print(f"\t\t-> Varianza Media de Vecinos: {(var*100.0):.4f}, Desviación Típica: {(std*10.0):.4f} puntos")
                acc_var += var
                acc_std += std

                if num_isolated_t > 0:
                    print(f"\t\t⚠️ Hay {num_isolated_t} ({num_isolated_t/self.graph.num_nodes*100:.2f}%) nodos aislados.")
                else:
                    print(f"\t\t✅ No hay nodos aislados.")
            
            metrics['dynamic_max_variance'] = max_var
            metrics['dynamic_min_variance'] = min_var
            metrics['dynamic_avg_variance'] = acc_var / len(self.graph.dynamic_edge_indices)
            print(f"\tPROMEDIO SEMANAL -> Varianza: {(acc_var/len(self.graph.dynamic_edge_indices)*100.0):.4f}, Desviación Típica: {(acc_std/len(self.graph.dynamic_edge_indices)*10.0):.4f} puntos")
        
        print("TEST DE HOMOFILIA GLOBAL:")
        # Calcular homofilia global
        homophily = self.calculate_grade_homophily()
        metrics['grade_homophily'] = homophily
        print(f"\t📊 Correlación (Homofilia) Notas-Vecinos: {homophily:.4f}")

        # Calcular asortatividad
        assort = self.calc_assortativity()
        metrics['assortativity'] = assort
        print(f"\t🔗 Asortatividad: {assort:.4f}")

        # Calcular energía de Dirichlet
        dirichlet_energy = self.calc_dirichlet_energy()
        metrics['dirichlet_energy'] = dirichlet_energy
        print(f"\t⚡ Energía de Dirichlet Media: {dirichlet_energy:.4f} (Más bajo = Mejor suavidad entre vecinos)")

        return metrics
        
    def test_all_graphs(self):
        """Función para testear todos los grafos en la carpeta GRAPH_DIR."""
        graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.pt')]
        all_metrics = {}
        
        for g in graphs:
            print(f"\n================= TESTEANDO GRAFO: {g} =================")
            self.graph_path = os.path.join(GRAPH_DIR, g)
            self.graph: Data = self.load_graph()
            metrics = self.test_graph(graph_path=self.graph_path)
            all_metrics[g] = metrics
        
        return all_metrics
        
    def calculate_neighbor_variance(self, week: int = None):
        """
        Calcula la varianza media de las notas de los vecinos para evaluar la calidad del grafo.
    
        Returns:
            avg_variance: La varianza media de todo el grafo.
        """
        
        # Aseguramos que y sea plano [N]
        y_flat = self.graph.y.squeeze()
        
        # Obtenemos origen (src) y destino (dst) de las aristas
        # src son los nodos para los que vamos a calcular la varianza de sus vecinos (dst)
        if not week:
            src, dst = self.graph.edge_index
        elif hasattr(self.graph, 'dynamic_edge_indices') and week < len(self.graph.dynamic_edge_indices):
            src, dst = self.graph.dynamic_edge_indices[week]
        else:
            raise ValueError("Semana inválida para grafo dinámico.")

        # Preparamos un tensor para guardar la varianza de cada alumno
        num_nodes = y_flat.shape[0]
        variances = []
        
        # --- BUCLE DE CÁLCULO ---
        # Nota: Como N es pequeño (alumnos < 500), un bucle es rápido y legible.
        # Si tuvieras millones de nodos, usaríamos scatter_reduce.
        
        count_valid_nodes = 0
        
        for i in range(num_nodes):
            # 1. Buscar quiénes son los vecinos del alumno 'i'
            # edge_index[0] == i busca todas las aristas que salen de i
            neighbor_indices = dst[src == i]
            
            # Si no tiene vecinos (o solo 1), la varianza no es fiable/calculable
            if len(neighbor_indices) < 2:
                continue
                
            # 2. Obtener las notas de esos vecinos
            neighbor_grades = y_flat[neighbor_indices]
            
            # 3. Calcular varianza de esas notas
            # unbiased=False para varianza poblacional (divide por N), True para muestral (N-1)
            var = torch.var(neighbor_grades, unbiased=True) 
            variances.append(var)
            count_valid_nodes += 1

        if len(variances) == 0:
            print(f"⚠️ No se pudo calcular varianza (¿sin vecinos?).")
            return float('inf')

        # 4. Promediar las varianzas de todos los alumnos
        variances_tensor = torch.stack(variances)
        avg_variance = torch.mean(variances_tensor).item()
        std_dev_neighbors = torch.sqrt(torch.tensor(avg_variance)).item() # Desviación típica promedio    
        
        return avg_variance, std_dev_neighbors
    
    def get_graph_outliers(self, threshold=0.3):
        """Devuelve IDs de alumnos cuya nota difiere mucho de la media de sus vecinos."""
        y_flat = self.graph.y.squeeze()
        src, dst = self.graph.edge_index
        outliers = []
        
        for i in range(y_flat.shape[0]):
            neighbors = dst[src == i]
            if len(neighbors) < 1: continue
            
            mean_neighbor_grade = torch.mean(y_flat[neighbors])
            my_grade = y_flat[i]
            
            diff = torch.abs(my_grade - mean_neighbor_grade)
            
            if diff > threshold:
                outliers.append(i)
                
        return outliers

    def calculate_grade_homophily(self):
        """Calcula la correlación entre nota del estudiante y nota media de sus vecinos."""
        src, dst = self.graph.edge_index
        y = self.graph.y.squeeze()
        
        # Calcular media de vecinos
        # scatter_mean requiere torch_scatter, si no tienes, usamos bucle o pandas
        # Versión simple con pandas para no complicar dependencias:
        
        df = pd.DataFrame({'src': src.cpu(), 'dst': dst.cpu(), 'grade_dst': y[dst].cpu()})
        
        # Agrupamos por nodo origen y calculamos la media de sus destinos
        neighbor_means = df.groupby('src')['grade_dst'].mean()
        
        # Alineamos con las notas reales
        # Ojo: puede haber nodos sin vecinos que no salgan en el groupby
        df_corr = pd.DataFrame({'my_grade': y.cpu().numpy()})
        df_corr['neighbor_mean'] = neighbor_means
        
        # Limpiamos NaNs (nodos aislados)
        df_corr = df_corr.dropna()
        
        correlation = df_corr['my_grade'].corr(df_corr['neighbor_mean'])
        return correlation

    def calc_assortativity(self):
        # Convertimos a NetworkX para usar sus métricas probadas
        G = to_networkx(self.graph, to_undirected=True)
        
        # Asignamos la nota como atributo al nodo
        y_np = self.graph.y.squeeze().cpu().numpy()
        attrs = {i: {'grade': val} for i, val in enumerate(y_np)}
        nx.set_node_attributes(G, attrs)
        
        # Calculamos asortatividad
        assortativity = nx.numeric_assortativity_coefficient(G, 'grade')
        return assortativity

    def calc_dirichlet_energy(self):
        src, dst = self.graph.edge_index
        y = self.graph.y.squeeze()
        
        # Diferencia al cuadrado entre vecinos
        squared_diff = (y[src] - y[dst]) ** 2
        energy = torch.sum(squared_diff).item()
        
        # Normalizar por número de aristas para que sea interpretable
        mean_energy = energy / self.graph.edge_index.shape[1]
        
        return mean_energy
    
    def _select_graph(self):
        """Función para listar los grafos guardados en la carpeta correspondiente y permitir al usuario seleccionar uno."""
        graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.pt')]
        if not graphs:
            print("No hay grafos guardados en la carpeta.")
            return None
        
        print("Grafos disponibles:")
        for i, g in enumerate(graphs):
            print(f"{i+1}. {g}")
        
        choice = int(input("Selecciona el número del grafo que quieres cargar: ")) - 1
        if 0 <= choice < len(graphs):
            return os.path.join(GRAPH_DIR, graphs[choice])
        else:
            print("Selección inválida.")
            return None

    def save_metrics(self, metrics: dict, output_path: str):
        """Guarda las métricas en un archivo CSV."""
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.to_csv(output_path)
        print(f"Métricas guardadas en {output_path}")
    
    ###################### VISUALIZACIÓN DE MÉTRICAS ######################
    # Función para extraer qué tipo de grafo es (Asistencia, Notas, Híbrido...)
    def parse_info(self,filename):
        # Formato esperado: Metodo_Datos_graph_KNN.pt
        # Ej: Temp_a_graph_5NN.pt
        clean = filename.replace('.pt', '')
        parts = clean.split('_')
        
        # --- A. ESTRATEGIA DE PROCESAMIENTO ---
        method_code = parts[0]
        method_map = {
            'Temp': 'Temporal\n(Secuencia completa)',
            'MP': 'Mean Pooling\n(Media semanal)',
            'Concat': 'Concatenation\n(Aplanado)',
            'LDS': 'LDS (Learnable)'
        }
        method_label = method_map.get(method_code, method_code)
        
        # --- B. FUENTE DE DATOS ---
        data_code = parts[1]
        data_map = {
            'a': 'Solo Asistencia',
            'g': 'Solo Notas',
            'ag': 'Híbrido (A+N)',
            'ga': 'Híbrido (A+N)',
            'f3w': 'Primeras 3 Semanas'
        }
        data_label = data_map.get(data_code, data_code)
        
        return pd.Series([method_label, data_label])
    
    # ==========================================
    # GRÁFICO DE HOMOFILIA (AGRUPAMIENTO)
    # ==========================================
    def plot_homophily(self, df, save_path= METRICS_MEDIA_DIR + 'grafico_homofilia.png'):
        """Genera el gráfico de Homofilia (Correlación Notas-Vecinos)."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.barplot(
            data=df,
            x='Datos',
            y='grade_homophily',
            hue='Estrategia',
            palette='viridis',
            edgecolor='black'
        )
        
        plt.title('Calidad de Agrupamiento (Homofilia)\n¿Los alumnos se conectan con compañeros de notas similares?', 
                fontsize=14, fontweight='bold')
        plt.ylabel('Correlación de Pearson (Mayor es mejor)', fontsize=12)
        plt.xlabel('Fuente de Datos', fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(title='Estrategia', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Gráfico guardado: {save_path}")
        plt.close() # Cierra la figura para liberar memoria

    # ==========================================
    # GRÁFICO DE SUAVIDAD (VARIANZA)
    # ==========================================
    def plot_smoothness(self, df, save_path= METRICS_MEDIA_DIR +'grafico_suavidad.png'):
        """Genera el gráfico de Dispersión de vecinos (Desviación Típica)."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.barplot(
            data=df,
            x='Datos',
            y='neighbor_std_dev', # Usamos Std Dev porque es más interpretable que Varianza
            hue='Estrategia',
            palette='magma',
            edgecolor='black'
        )
        
        plt.title('Estabilidad de la Señal en el Grafo\n¿Cuánta diferencia de nota hay entre un alumno y sus vecinos?', 
                fontsize=14, fontweight='bold')
        plt.ylabel('Desviación Típica (Menor es mejor)', fontsize=12)
        plt.xlabel('Fuente de Datos', fontsize=12)
        plt.legend(title='Estrategia', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Gráfico guardado: {save_path}")
        plt.close()

    # ==========================================
    # GRÁFICO DE RUIDO (OUTLIERS)
    # ==========================================
    def plot_outliers(self, df, save_path= METRICS_MEDIA_DIR +'grafico_outliers.png'):
        """Genera el gráfico de Nodos Atípicos (Outliers)."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.barplot(
            data=df,
            x='Datos',
            y='num_outliers',
            hue='Estrategia',
            palette='Reds',
            edgecolor='black'
        )
        
        plt.title('Nodos Atípicos (Ruido)\nCantidad de alumnos mal conectados', 
                fontsize=14, fontweight='bold')
        plt.ylabel('Número de Alumnos (Menor es mejor)', fontsize=12)
        plt.xlabel('Fuente de Datos', fontsize=12)
        plt.legend(title='Estrategia', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Gráfico guardado: {save_path}")
        plt.close()
    
    # ==========================================
    # GRÁFICO DE ASORTATIVIDAD
    # ==========================================
    def plot_assortativity(self, df, save_path= METRICS_MEDIA_DIR +'grafico_asortatividad.png'):
        """
        Genera el gráfico de Asortatividad Numérica.
        Mide si los nodos con valores altos se conectan con otros altos.
        """
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.barplot(
            data=df,
            x='Datos',
            y='assortativity',
            hue='Estrategia',
            palette='plasma', # Usamos una paleta distinta para diferenciar
            edgecolor='black'
        )
        
        plt.title('Asortatividad del Grafo\n¿Se conectan los estudiantes con perfiles idénticos?', 
                fontsize=14, fontweight='bold')
        plt.ylabel('Coeficiente de Asortatividad (Cercano a 1 es mejor)', fontsize=12)
        plt.xlabel('Fuente de Datos', fontsize=12)
        plt.ylim(0, 1.05) # Rango teórico -1 a 1, pero esperamos positivo
        plt.legend(title='Estrategia', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Gráfico guardado: {save_path}")
        plt.close()

    # ==========================================
    # GRÁFICO DE ENERGÍA DE DIRICHLET
    # ==========================================
    def plot_dirichlet(self,df, save_path=METRICS_MEDIA_DIR+'grafico_energia.png'):
        """
        Genera el gráfico de Energía de Dirichlet.
        Mide la 'suavidad' global. Cuanto más bajo, más suave es la transición de notas.
        """
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.barplot(
            data=df,
            x='Datos',
            y='dirichlet_energy',
            hue='Estrategia',
            palette='coolwarm',
            edgecolor='black'
        )
        
        plt.title('Energía de Dirichlet (Suavidad Global)\n¿Cuán bruscos son los cambios de nota en el grafo?', 
                fontsize=14, fontweight='bold')
        plt.ylabel('Energía Media (Menor es mejor)', fontsize=12)
        plt.xlabel('Fuente de Datos', fontsize=12)
        plt.legend(title='Estrategia', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Gráfico guardado: {save_path}")
        plt.close()

    def visualize_metrics(self, metrics_path: str):
        """Genera gráficos a partir del archivo de métricas guardado."""
        
        # 1. CARGAR DATOS
        df_metrics = pd.read_csv(metrics_path)

        # Renombrar la primera columna si no tiene nombre
        if 'Unnamed: 0' in df_metrics.columns:
            df_metrics = df_metrics.rename(columns={'Unnamed: 0': 'Filename'})
        elif df_metrics.columns[0] != 'Filename':
            df_metrics.columns.values[0] = 'Filename'

        
        df_metrics[['Estrategia', 'Datos']] = df_metrics['Filename'].apply(self.parse_info)

        
        print("Datos procesados:")
        print(df_metrics[['Estrategia', 'Datos', 'grade_homophily']].head())
        
        
        self.plot_homophily(df_metrics)
        self.plot_smoothness(df_metrics)
        self.plot_outliers(df_metrics)
        self.plot_assortativity(df_metrics)
        self.plot_dirichlet(df_metrics)
        
        print("\n🎉 ¡Proceso completado! Tienes 5 nuevas imágenes.")
        

if __name__== "__main__":
    
    tester = GraphTester()
    metrics = tester.test_all_graphs()
    path_metrics = "data/metrics/graph_metrics.csv"
    tester.save_metrics(metrics, path_metrics)
    tester.visualize_metrics(path_metrics)