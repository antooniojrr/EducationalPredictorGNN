import os
import numpy as np
# ---------------------------------------------------------
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------------------------------------
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


GRAPH_DIR = './data/processed/graphs/'
METRICS_MEDIA_DIR = './media/graphMetrics/'
os.makedirs(METRICS_MEDIA_DIR, exist_ok=True)


class GraphTester:
    """
    Clase encargada de evaluar la calidad estructural y estadística
    de los grafos académicos generados.

    Proporciona métricas de homofilia, suavidad, asortatividad,
    energía de Dirichlet y conectividad algebraica, así como
    herramientas de visualización comparativa.
    """        

    def load_graph(self):
        """
        Carga en memoria el grafo especificado en self.graph_path.

        Returns:
            Data: Objeto Data previamente almacenado.

        Raises:
            FileNotFoundError: Si el archivo no existe.
        """
        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"El archivo {self.graph_path} no existe.")
        
        g = torch.load(self.graph_path, weights_only=False)
        return g


    def test_graph(self, graph_path: str = None):
        """
        Ejecuta un conjunto completo de pruebas estructurales
        y estadísticas sobre un grafo.

        Evalúa:
            - Nodos aislados.
            - Varianza de etiquetas entre vecinos.
            - Outliers estructurales.
            - Métricas dinámicas (si aplica).
            - Homofilia.
            - Asortatividad.
            - Energía de Dirichlet.
            - Conectividad algebraica.

        Args:
            graph_path (str, optional): Ruta del grafo a evaluar.

        Returns:
            tuple:
                dict: Métricas globales calculadas.
                list or None: Varianza semanal si el grafo es dinámico.
        """
        while graph_path is None:
            graph_path = self._select_graph()
            
        self.graph_path = graph_path
        self.graph: Data = self.load_graph()
        
        metrics = {}
        print("Iniciando evaluación del grafo especificado...")

        connected_nodes = torch.unique(self.graph.edge_index)
        num_connected = connected_nodes.size(0)
        num_isolated = self.graph.num_nodes - num_connected
        metrics['num_isolated_nodes'] = num_isolated

        var, std = self.calc_neighbor_variance()
        metrics['neighbor_variance'] = var
        metrics['neighbor_std_dev'] = std

        outliers = self.get_graph_outliers(threshold=0.3)
        metrics['num_outliers'] = len(outliers)

        var_semanal = None
        if hasattr(self.graph, 'dynamic_edge_indices'):
            var_semanal = []
            acc_var = 0.0
            acc_std = 0.0
            max_var = -float('inf')
            min_var = float('inf')

            for t, edge_index_t in enumerate(self.graph.dynamic_edge_indices):
                var, std = self.calc_neighbor_variance(week=t)
                var_semanal.append(var)

                max_var = max(max_var, var)
                min_var = min(min_var, var)

                acc_var += var
                acc_std += std

            metrics['dynamic_max_variance'] = max_var
            metrics['dynamic_min_variance'] = min_var
            metrics['dynamic_avg_variance'] = acc_var / len(self.graph.dynamic_edge_indices)

        homophily = self.calc_grade_homophily()
        metrics['grade_homophily'] = homophily

        assort = self.calc_assortativity()
        metrics['assortativity'] = assort

        dirichlet_energy = self.calc_dirichlet_energy()
        metrics['dirichlet_energy'] = dirichlet_energy

        algebraic_connectivity = self.calc_algebraic_connectivity()
        metrics['algebraic_connectivity'] = algebraic_connectivity

        if var_semanal:
            return metrics, var_semanal
        else:
            return metrics, None
        

    def test_all_graphs(self):
        """
        Evalúa todos los grafos almacenados en GRAPH_DIR.

        Returns:
            tuple:
                dict: Métricas por grafo.
                dict: Varianza semanal por grafo dinámico.
        """
        graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.pt')]
        all_metrics = {}
        all_var_semanal = {}
        
        for g in graphs:
            self.graph_path = os.path.join(GRAPH_DIR, g)
            self.graph: Data = self.load_graph()
            metrics, var_semanal = self.test_graph(graph_path=self.graph_path)

            if var_semanal:
                all_var_semanal[g] = var_semanal

            all_metrics[g] = metrics
        
        return all_metrics, all_var_semanal
        

    def calc_neighbor_variance(self, week: int = None):
        """
        Calcula la varianza media de las etiquetas de los vecinos
        para cada nodo y devuelve su promedio global.

        Args:
            week (int, optional): Semana específica en grafos dinámicos.

        Returns:
            tuple:
                float: Varianza media.
                float: Desviación típica promedio.
        """
        y_flat = self.graph.y.squeeze()

        if not week:
            src, dst = self.graph.edge_index
        elif hasattr(self.graph, 'dynamic_edge_indices') and week < len(self.graph.dynamic_edge_indices):
            src, dst = self.graph.dynamic_edge_indices[week]
        else:
            raise ValueError("Semana inválida para grafo dinámico.")

        num_nodes = y_flat.shape[0]
        variances = []

        for i in range(num_nodes):
            neighbor_indices = dst[src == i]

            if len(neighbor_indices) < 2:
                continue
                
            neighbor_grades = y_flat[neighbor_indices]
            var = torch.var(neighbor_grades, unbiased=True)
            variances.append(var)

        if len(variances) == 0:
            return float('inf')

        variances_tensor = torch.stack(variances)
        avg_variance = torch.mean(variances_tensor).item()
        std_dev_neighbors = torch.sqrt(torch.tensor(avg_variance)).item()
        
        return avg_variance, std_dev_neighbors
        

    def get_graph_outliers(self, threshold=0.3):
        """
        Identifica nodos cuya etiqueta difiere significativamente
        de la media de sus vecinos.

        Args:
            threshold (float): Diferencia mínima considerada anómala.

        Returns:
            list: Índices de nodos atípicos.
        """
        y_flat = self.graph.y.squeeze()
        src, dst = self.graph.edge_index
        outliers = []
        
        for i in range(y_flat.shape[0]):
            neighbors = dst[src == i]
            if len(neighbors) < 1:
                continue
            
            mean_neighbor_grade = torch.mean(y_flat[neighbors])
            my_grade = y_flat[i]
            
            diff = torch.abs(my_grade - mean_neighbor_grade)
            
            if diff > threshold:
                outliers.append(i)
                
        return outliers


    def calc_grade_homophily(self):
        """
        Calcula la correlación de Pearson entre la nota
        de cada estudiante y la media de sus vecinos.

        Returns:
            float: Coeficiente de correlación.
        """
        src, dst = self.graph.edge_index
        y = self.graph.y.squeeze()

        df = pd.DataFrame({'src': src.cpu(), 'dst': dst.cpu(), 'grade_dst': y[dst].cpu()})
        neighbor_means = df.groupby('src')['grade_dst'].mean()

        df_corr = pd.DataFrame({'my_grade': y.cpu().numpy()})
        df_corr['neighbor_mean'] = neighbor_means
        df_corr = df_corr.dropna()

        correlation = df_corr['my_grade'].corr(df_corr['neighbor_mean'])
        return correlation


    def calc_assortativity(self):
        """
        Calcula el coeficiente de asortatividad numérica
        respecto a la nota final.

        Returns:
            float: Coeficiente de asortatividad.
        """
        G = to_networkx(self.graph, to_undirected=True)
        
        y_np = self.graph.y.squeeze().cpu().numpy()
        attrs = {i: {'grade': val} for i, val in enumerate(y_np)}
        nx.set_node_attributes(G, attrs)
        
        assortativity = nx.numeric_assortativity_coefficient(G, 'grade')
        return assortativity


    def calc_dirichlet_energy(self):
        """
        Calcula la energía de Dirichlet media del grafo,
        que mide la suavidad global de la señal sobre la estructura.

        Returns:
            float: Energía media normalizada.
        """
        src, dst = self.graph.edge_index
        y = self.graph.y.squeeze()
        
        squared_diff = (y[src] - y[dst]) ** 2
        energy = torch.sum(squared_diff).item()
        mean_energy = energy / self.graph.edge_index.shape[1]
        
        return mean_energy
    

    def calc_algebraic_connectivity(self):
        """
        Calcula la conectividad algebraica del grafo
        (segundo autovalor más pequeño del Laplaciano).

        Returns:
            float: Conectividad algebraica.
        """
        G = to_networkx(self.graph, to_undirected=True)
        laplacian_matrix = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian_matrix)

        if len(eigenvalues) > 1:
            return eigenvalues[1]
        else:
            return 0.0


    def _select_graph(self):
        """
        Permite seleccionar interactívamente un grafo
        disponible en el directorio correspondiente.

        Returns:
            str or None: Ruta del grafo seleccionado.
        """
        graphs = [f for f in os.listdir(GRAPH_DIR) if f.endswith('.pt')]
        if not graphs:
            return None
        
        for i, g in enumerate(graphs):
            print(f"{i+1}. {g}")  # listado de grafos disponibles
        
        choice = int(input("Seleccione el número correspondiente al grafo que desea cargar: ")) - 1
        if 0 <= choice < len(graphs):
            return os.path.join(GRAPH_DIR, graphs[choice])
        else:
            return None


    def save_metrics(self, metrics: dict, output_path: str):
        """
        Guarda las métricas calculadas en un archivo CSV.

        Args:
            metrics (dict): Diccionario de métricas.
            output_path (str): Ruta de destino.
        """
        df = pd.DataFrame.from_dict(metrics, orient='index')
        df.to_csv(output_path)


    def visualize_metrics(self, metrics_path: str):
        """
        Genera automáticamente todos los gráficos comparativos
        a partir del archivo CSV de métricas.

        Args:
            metrics_path (str): Ruta del archivo de métricas.
        """
        df_metrics = pd.read_csv(metrics_path)

        if 'Unnamed: 0' in df_metrics.columns:
            df_metrics = df_metrics.rename(columns={'Unnamed: 0': 'Filename'})
        elif df_metrics.columns[0] != 'Filename':
            df_metrics.columns.values[0] = 'Filename'

        df_metrics[['Estrategia', 'Datos']] = df_metrics['Filename'].apply(self.parse_info)

        self.plot_homophily(df_metrics)
        self.plot_smoothness(df_metrics)
        self.plot_outliers(df_metrics)
        self.plot_assortativity(df_metrics)
        self.plot_dirichlet(df_metrics)
        self.plot_algebraic_connectivity(df_metrics)
        self.plot_temporal_variance(df_metrics)


    def visualize_temporal_variance(self, var_semanal_dict):
        """
        Genera un gráfico comparativo de la evolución temporal
        de la varianza de vecinos para grafos dinámicos.

        Args:
            var_semanal_dict (dict): Varianzas semanales por grafo.
        """
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        for graph_name, var_semanal in var_semanal_dict.items():
            if var_semanal:
                name = self.parse_info(graph_name)[1]
                weeks = list(range(len(var_semanal)))
                plt.plot(weeks, var_semanal, marker='o', label=name)
        
        plt.title('Evolución de la Varianza de Vecinos en Grafos Dinámicos')
        plt.ylabel('Varianza Media de Vecinos')
        plt.xlabel('Semana')
        plt.legend(title='Grafo', loc='upper right')
        plt.tight_layout()
        save_path = METRICS_MEDIA_DIR + 'grafico_temporal_variance.png'
        plt.savefig(save_path, dpi=300)
        plt.close()

if __name__== "__main__":
    
    tester = GraphTester()
    metrics, var_semanal = tester.test_all_graphs()
    path_metrics = "data/metrics/graph_metrics.csv"
    tester.save_metrics(metrics, path_metrics)
    tester.visualize_metrics(path_metrics)
    tester.visualize_temporal_variance(var_semanal)