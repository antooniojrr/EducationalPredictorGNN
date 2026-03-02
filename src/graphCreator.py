import os
# ---------------------------------------------------------
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
# ---------------------------------------------------------
from dataLoader import DataLoader

"""
Módulo responsable de la construcción de grafos académicos a partir
del dataset original.

Cada estudiante se modela como un nodo y las aristas se generan
mediante un algoritmo k-NN en función de distintos perfiles
de similitud académica (asistencia, notas, encuestas, etc.).

Permite generar grafos estáticos o dinámicos (temporales) y
almacenarlos para su reutilización.
"""

class GraphCreator:
    """
    Clase encargada de construir grafos académicos en formato
    compatible con PyTorch Geometric.

    Ofrece:
        - Generación de grafos estáticos y dinámicos.
        - Diferentes perfiles de similitud.
        - Persistencia en disco para evitar recomputación.
    """

    SIM_PROFILES = ['a', 'g', 'a&g', 'f3w', 'all', 'surveys']

    def __init__(self, semana_final: int = None):
        """
        Inicializa el generador de grafos.

        Args:
            semana_final (int, optional): Si se especifica, limita
                la construcción del grafo a datos disponibles hasta
                dicha semana (escenario de predicción parcial).
        """
        self.num_weeks = semana_final
        self.data_loader = None
        if semana_final is not None:
            self.data_loader = DataLoader(num_weeks=semana_final)
        else:
            self.data_loader = DataLoader()
    

    def create_graph(self, cat_opt=None, sim_profile: str = 'a&g', k_neighbors=5, dyn_graph=True) -> Data:
        """
        Construye un grafo académico a partir del dataset cargado.

        El procedimiento incluye:
            - Carga de datos y componentes crudos.
            - Construcción de la matriz de adyacencia mediante k-NN.
            - Creación del objeto Data de PyG.
            - Generación opcional de grafos dinámicos por semana.
            - Almacenamiento en disco del grafo procesado.

        Args:
            cat_opt (str): Opción de categorización del dataset.
            sim_profile (str): Perfil de similitud utilizado.
            k_neighbors (int): Número de vecinos en el grafo k-NN.
            dyn_graph (bool): Indica si se construye un grafo dinámico.

        Returns:
            Data: Objeto Data de PyTorch Geometric.
        """
        assert cat_opt in self.data_loader.CAT_OPTIONS or None, \
            "cat_opt debe ser una opción válida o None para selección manual."
        
        cat_opt, self.X, self.Y, raw_comps = self.data_loader.load_data(cat_opt=cat_opt)
        
        edge_index = self.create_adj_matrix(sim_profile, raw_comps, k=k_neighbors)

        self.data = Data(x=self.X, edge_index=edge_index, y=self.Y)

        if dyn_graph and cat_opt == 'Temp' and sim_profile not in ['f3w', 'g']: 
            print("Generando grafo dinámico por semana...")
            dynamic_edge_indices = []
        
            for t in range(self.X.shape[1]):  
                edge_index_t = self.create_adj_matrix(sim_profile=sim_profile, raw_comps=raw_comps, k=k_neighbors, t=t)
                dynamic_edge_indices.append(edge_index_t)
            
            self.data.dynamic_edge_indices = dynamic_edge_indices  
        
        path = f"./data/processed/graphs/{cat_opt}_{sim_profile}_"

        if dyn_graph:
            path = path + "dynamic_"
        
        if self.num_weeks:
            path = path + f"{self.num_weeks}weeks_"

        path = path + f"graph_{k_neighbors}NN.pt"

        torch.save(self.data, path)
        print(f"Grafo guardado en: {path}")
        return self.data


    def create_adj_matrix(self, sim_profile: str, raw_comps: list[torch.Tensor], k: int, t: int = None) -> torch.Tensor:
        """
        Construye la matriz de adyacencia del grafo mediante un
        enfoque k-NN basado en un perfil de similitud específico.

        Dependiendo del perfil seleccionado, se utilizan distintas
        combinaciones de características (asistencia, notas,
        encuestas, primeras semanas, semana concreta, etc.).

        Args:
            sim_profile (str): Perfil de similitud a emplear.
            raw_comps (list[Tensor]): Componentes crudos de las
                características del dataset.
            k (int): Número de vecinos a conectar por nodo.
            t (int, optional): Semana límite considerada en el
                cálculo de similitud (para grafos dinámicos).

        Returns:
            torch.Tensor: edge_index con las aristas del grafo.
        """
        metric = 'euclidean'

        match sim_profile:
            
            case 'a&g':
                att_feat = raw_comps[0].squeeze(-1)
                if t and t < att_feat.shape[1]:
                    att_feat = att_feat[:, :t+1]

                grades_flat = raw_comps[1].squeeze(-1)
                if t and t < grades_flat.shape[1]:
                    grades_flat = grades_flat[:, :t+1]

                week_sum = grades_flat.sum(dim=0)
                active_weeks = torch.nonzero(week_sum > 0).squeeze()

                if active_weeks.numel() > 0:
                    grades_feat = grades_flat[:, active_weeks]
                    if grades_feat.dim() == 1:
                        grades_feat = grades_feat.unsqueeze(1)
                else:
                    grades_feat = torch.empty((grades_flat.shape[0], 0))

                similarity_profile = torch.cat([att_feat, grades_feat], dim=1).numpy()

            case 'a':
                att_feat = raw_comps[0].squeeze(-1)
                if t and t < att_feat.shape[1]:
                    att_feat = att_feat[:, :t+1]

                similarity_profile = torch.cat([att_feat], dim=1).numpy()
                metric = 'hamming'
            
            case 'g':
                grades_flat = raw_comps[1].squeeze(-1)

                if t and t < grades_flat.shape[1]:
                    grades_flat = grades_flat[:, :t+1]

                week_sum = grades_flat.sum(dim=0)
                active_weeks = torch.nonzero(week_sum > 0).squeeze()
                grades_feat = grades_flat[:, active_weeks]

                similarity_profile = torch.cat([grades_feat], dim=1).numpy()
            
            case 'f3w':
                similarity_profile = torch.empty((len(raw_comps[0]), 0))
                if raw_comps[0].ndim != 3 or raw_comps[0].shape[1] < 3:
                    raise ValueError("Se requieren al menos 3 semanas para el perfil 'f3w'.")
                for feat in raw_comps:
                    if feat.ndim == 3:
                        feat_flat = feat[:, 0:3, :].reshape(len(feat), -1) 
                        similarity_profile = torch.cat([similarity_profile, feat_flat], dim=1)
                        
            case 'week':
                if t is None or t > raw_comps[0].shape[1]:
                    raise ValueError("Debe especificarse una semana válida para el perfil 'week'.")
                
                similarity_profile = torch.empty((len(raw_comps[0]), 0))
                for feat in raw_comps:
                    if feat.ndim == 3:
                        feat_t = feat[:, t, :]
                        similarity_profile = torch.cat([similarity_profile, feat_t], dim=1)
            
            case 'all':
                att_feat = raw_comps[0].squeeze(-1)
                if t and t < att_feat.shape[1]:
                    att_feat = att_feat[:, :t+1]

                grades_flat = raw_comps[1].squeeze(-1)
                if t and t < grades_flat.shape[1]:
                    grades_flat = grades_flat[:, :t+1]

                week_sum = grades_flat.sum(dim=0)
                active_weeks = torch.nonzero(week_sum > 0).squeeze()

                if active_weeks.numel() > 0:
                    grades_feat = grades_flat[:, active_weeks]
                    if grades_feat.dim() == 1:
                        grades_feat = grades_feat.unsqueeze(1)
                else:
                    grades_feat = torch.empty((grades_flat.shape[0], 0))
                
                surveys_feat = raw_comps[2]
                if t and t < surveys_feat.shape[1]:
                    surveys_feat = surveys_feat[:, :t+1, :]
                
                surveys_flat = surveys_feat.reshape(surveys_feat.shape[0], -1)
                similarity_profile = torch.cat([att_feat, grades_feat, surveys_flat], dim=1).numpy()

            case 'surveys':
                surveys_feat = raw_comps[2]
                if t and t < surveys_feat.shape[1]:
                    surveys_feat = surveys_feat[:, :t+1, :]
                surveys_flat = surveys_feat.reshape(surveys_feat.shape[0], -1)
                similarity_profile = surveys_flat.numpy()

            case 'default':
                raise ValueError("Perfil de similitud no reconocido.")
        
        adj_matrix = kneighbors_graph(similarity_profile, k, mode='connectivity', include_self=False, metric=metric)
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)

        return edge_index


    def create_all_graphs(self):
        """
        Genera y almacena en disco todos los grafos posibles
        combinando perfiles de similitud y opciones de categorización.
        """
        for profile in self.SIM_PROFILES:
            for cat_opt in self.data_loader.CAT_OPTIONS:
                if cat_opt == 'Temp':
                    print(f"\nIniciando creación de grafo para cat_opt: {cat_opt} (dinámico = False) y perfil: {profile}...")
                    self.create_graph(cat_opt=cat_opt, sim_profile=profile, dyn_graph=False)
                    print(f"\nIniciando creación de grafo para cat_opt: {cat_opt} (dinámico = True) y perfil: {profile}...")
                    self.create_graph(cat_opt=cat_opt, sim_profile=profile, dyn_graph=True)
                else:
                    print(f"\nIniciando creación de grafo para cat_opt: {cat_opt} y perfil: {profile}...")
                    self.create_graph(cat_opt=cat_opt, sim_profile=profile, dyn_graph=False)


    def load_graph(self, cat_opt: str = 'Temp', sim_profile: str = 'a&g', k_neighbors: int = 5, dyn_graph: bool = True) -> Data:
        """
        Carga un grafo previamente almacenado en disco.

        Si el archivo no existe, el grafo se genera automáticamente
        con la configuración especificada.

        Args:
            cat_opt (str): Tipo de categorización del dataset.
            sim_profile (str): Perfil de similitud.
            k_neighbors (int): Número de vecinos en el grafo.
            dyn_graph (bool): Indica si el grafo es dinámico.

        Returns:
            Data: Objeto Data cargado o recién generado.
        """
        path = f"./data/processed/graphs/{cat_opt}_{sim_profile}_"
        if dyn_graph:
            path = path + "dynamic_"
        path = path + f"graph_{k_neighbors}NN.pt"

        if not os.path.exists(path):
            g = self.create_graph(cat_opt=cat_opt, sim_profile=sim_profile, k_neighbors=k_neighbors, dyn_graph=dyn_graph)
        else: 
            g = torch.load(path, weights_only=False)

        return g
    

    def get_features_names(self) -> list[str]:
        """
        Devuelve los nombres de las características utilizadas
        en el dataset actual.
        """
        return self.data_loader.FEATURE_NAMES
    

if __name__ == "__main__":
    """
    Ejecución directa del módulo para generar y almacenar
    todos los grafos definidos.
    """
    loader = GraphCreator()
    loader.create_all_graphs()

    print("Grafos creados y guardados.")