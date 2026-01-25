import os
import pandas as pd
import torch
import numpy as np
import datetime
import re
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

from dataLoader import DataLoader

"""
Clase para cargar de mi DS en concreto los datos en un grafo temporal, con estudiantes como nodos y matriz de adyacencia basada en similitud académica
y calculada por algoritmo k-NN.

Su método principal será create_graph(), que devolverá un objeto Data de PyG con el grafo temporal de estudiantes.
"""
class GraphCreator:
    SIM_PROFILES = ['a', 'g', 'a&g', 'f3w']

    def __init__(self, dl: DataLoader = DataLoader()):
        """
        dl: Instancia de DataLoader para cargar los datos base.
        """

        self.data_loader = dl
  
    
    def create_graph(self,cat_opt=None, sim_profile: str = 'a&g', k_neighbors=5, dyn_graph=True) -> Data:
        
        assert cat_opt in self.data_loader.CAT_OPTIONS or None, \
            "cat_opt debe ser una opción válida o None para selección manual."
        
        cat_opt,self.X, self.Y, raw_comps = self.data_loader.load_data(cat_opt=cat_opt)
        
        # CONSTRUIR GRAFO (Matriz de Adyacencia)
        edge_index = self.create_adj_matrix(sim_profile, raw_comps, k=k_neighbors)

        # Crear objeto Data de PyG
        self.data = Data(x=self.X, edge_index=edge_index, y=self.Y)

        # Si es dinámico, crear lista de edge_index por semana
        if dyn_graph and cat_opt == 'Temp':
            print(">>> Generando grafo dinámico semanal...")
            dynamic_edge_indices = []
        
            for t in range(self.X.shape[1]):  
                edge_index_t = self.create_adj_matrix(sim_profile='week', raw_comps=raw_comps, k=k_neighbors, t=t)
                
                dynamic_edge_indices.append(edge_index_t)
            
            self.data.dynamic_edge_indices = dynamic_edge_indices  # Lista de edge_index por semana
           
        
        # Guardar procesado para no recargar siempre
        path = f"./data/processed/graphs/{cat_opt}_{sim_profile}_"

        if dyn_graph:
            path = path + "dynamic_"
        
        path = path + f"graph_{k_neighbors}NN.pt"

        torch.save(self.data, path)
        print(f">>> Grafo guardado en: {path}")
        return self.data

    def create_adj_matrix(self, sim_profile: str, raw_comps: list[torch.Tensor], k: int, t: int = None) -> torch.Tensor:
        """
        Crea la matriz de adyacencia basada en el perfil de similitud dado.
        
        sim_profile: Perfil de similitud a usar ('a', 'g', 'a&g', 'f3w', etc.)
        k: Número de vecinos para conectar en el grafo (similitud)
        raw_comps: Componentes crudas de features de los nodos (alumnos)
        t: Si sim_profile es 'week', indica la semana a usar.
        
        Returns:
            edge_index: Tensor con los índices de las aristas del grafo.
        """
        metric = 'euclidean'
        match sim_profile:
            
            case 'a&g':
                # Obtengo la asistencia y solo las notas de las semanas en las que hubo alguna entrega
                att_feat = raw_comps[0].squeeze(-1)  # Asistencia
                
                grades_flat = raw_comps[1].squeeze(-1)  # Notas prácticas
                week_sum = grades_flat.sum(dim=0)
                active_weeks = torch.nonzero(week_sum>0).squeeze()

                grades_feat = grades_flat[:, active_weeks]/10.0  # Filtrar solo semanas activas

                similarity_profile = torch.cat([att_feat, grades_feat], dim=1).numpy()  # Concatenar asistencia y notas prácticas

            case 'a':
                att_feat = raw_comps[0].squeeze(-1)  # Asistencia
                similarity_profile = torch.cat([att_feat], dim=1).numpy()
                metric = 'hamming'  # Métrica Hamming para datos binarios
            
            case 'g':
                grades_flat = raw_comps[1].squeeze(-1)  # Notas prácticas
                week_sum = grades_flat.sum(dim=0)
                active_weeks = torch.nonzero(week_sum>0).squeeze()

                grades_feat = grades_flat[:, active_weeks]/10.0  # Filtrar solo semanas activas

                similarity_profile = torch.cat([grades_feat], dim=1).numpy()
            
            case 'f3w':
                # Usamos las features de las primeras semana para definir similitud inicial a partir de los raw_comps
                similarity_profile = torch.empty((len(raw_comps[0]), 0))
                for feat in raw_comps:
                    if feat.ndim == 3:
                        feat_flat = feat[:, 0:3, :].reshape(len(feat), -1) 
                        similarity_profile = torch.cat([similarity_profile, feat_flat], dim=1)
                        
            case 'week':
                if t is None:
                    raise ValueError("Debes especificar la semana 't' para el perfil de similitud 'week'.")
                
                similarity_profile = torch.empty((len(raw_comps[0]), 0))
                for feat in raw_comps:
                    if feat.ndim == 3:
                        feat_t = feat[:, t, :]
                        similarity_profile = torch.cat([similarity_profile, feat_t], dim=1)

            case 'default':
                raise ValueError("Perfil de similitud no reconocido.")
        
        adj_matrix = kneighbors_graph(similarity_profile, k, mode='connectivity', include_self=False, metric=metric)
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)

        return edge_index

    def create_all_graphs(self):
        """
        Crea y guarda grafos para todos los perfiles de similitud definidos en SIM_PROFILES y todas las opciones de CAT_OPTIONS.
        """
        for profile in self.SIM_PROFILES:
            for cat_opt in self.data_loader.CAT_OPTIONS:
                if cat_opt == 'Temp':
                    print(f"\nCreando grafo para cat_opt: {cat_opt} (Dynamic = False) y perfil de similitud: {profile}...")
                    self.create_graph(cat_opt=cat_opt, sim_profile=profile, dyn_graph=False)
                    print(f"\nCreando grafo para cat_opt: {cat_opt} (Dynamic = True) y perfil de similitud: {profile}...")
                    self.create_graph(cat_opt=cat_opt, sim_profile=profile, dyn_graph=True)
                else:
                    print(f"\nCreando grafo para cat_opt: {cat_opt} y perfil de similitud: {profile}...")
                    self.create_graph(cat_opt=cat_opt, sim_profile=profile, dyn_graph=False)

    def load_graph(self, cat_opt: str = 'Temp', sim_profile: str = 'a&g', k_neighbors: int = 5, dyn_graph: bool = True) -> Data:
        """Carga el grafo desde el archivo especificado."""
        path = f"./data/processed/graphs/{cat_opt}_{sim_profile}_"
        if dyn_graph:
            path = path + "dynamic_"
        path = path + f"graph_{k_neighbors}NN.pt"

        if not os.path.exists(path):
            g = self.create_graph(cat_opt=cat_opt, sim_profile=sim_profile, k_neighbors=k_neighbors, dyn_graph=dyn_graph)
        else: 
            g = torch.load(path, weights_only=False)
        #print(f"Grafo cargado con {g.num_nodes} nodos y {g.num_edges} aristas.")
        return g
    
# --- Para testeo ---
if __name__ == "__main__":
    # Ajusta la ruta a tu carpeta de datos
    loader = GraphCreator()
    loader.create_all_graphs()

    print("Grafos creados y guardados.")
    