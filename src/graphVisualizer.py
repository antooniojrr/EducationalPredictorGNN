import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import os
import imageio.v2 as imageio

GRAPH_DIR = './data/processed/graphs/'
GRAPH_MEDIA_DIR = './media/graph_visualizations/'
GRAPH_FRAMES_DIR = GRAPH_MEDIA_DIR + 'graph_frames/'
os.makedirs(GRAPH_MEDIA_DIR, exist_ok=True)

class GraphVisualizer:

    def visualize_static(self, path_grafo: str, week: int = None, name: str = 'grafo', show: bool = True):
        """
        Visualiza un grafo estático desde un archivo .pt.

        """
        if not os.path.exists(path_grafo):
            print(f"❌ No encuentro el archivo en: {path_grafo}")
            print("Ejecuta primero dataset.py para generarlo.")
            exit()

        print(f"Cargando grafo desde {path_grafo}...")
        data = torch.load(path_grafo, weights_only=False)

        if week is not None and hasattr(data, 'dynamic_edge_indices') and week < len(data.dynamic_edge_indices):
            print(f"Usando la estructura del grafo en la semana {week}...")
            data.edge_index = data.dynamic_edge_indices[week]

        # 2. CONVERTIR A NETWORKX
        # to_networkx transforma el tensor edge_index en un objeto grafo manejable
        G = to_networkx(data, to_undirected=True)

        # 3. CONFIGURAR EL DIBUJO
        plt.figure(figsize=(12, 8))

        # --- A) COLOREAR POR NOTA FINAL ---
        # data.y tiene las notas normalizadas (0 a 1). 
        # Vamos a usarlas para el color.
        node_colors = data.y.numpy().flatten()

        # --- B) CALCULAR POSICIONES (LAYOUT) ---
        # Spring Layout simula muelles. Nodos conectados se atraen.
        # k ajusta la distancia entre nodos.
        pos = nx.spring_layout(G, seed=42, k=0.3) 

        # --- C) DIBUJAR ---
        # Nodos
        nodes = nx.draw_networkx_nodes(G, pos, 
                            node_color=node_colors, 
                            cmap=plt.cm.RdYlGn, # Mapa de color: Rojo-Amarillo-Verde
                            node_size=300,
                            alpha=0.9)

        # Aristas (Las pintamos gris suave para no ensuciar)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

        # Etiquetas (IDs de alumnos)
        # Como los IDs reales eran 1, 2, 3... y el grafo empieza en 0, sumamos 1 para visualizar
        labels = {i: str(i+1) for i in range(data.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

        # Barra de color (Legend)
        plt.colorbar(nodes, label='Nota Final Normalizada (0=0, 1=10)')
        plt.title(f'Grafo de Estudiantes (k-NN Similitud)\nColor = Rendimiento Académico', fontsize=15)
        plt.axis('off')

        # 4. GUARDAR Y MOSTRAR
        output_file = GRAPH_MEDIA_DIR + f'{name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Imagen guardada en: {output_file}")

        if show:
            plt.show()

    def visualize_dynamic(self, path_grafo: str, name: str = 'grafo_dinamico', clean_frames: bool = True):
        if not os.path.exists(path_grafo):
            print("❌ Error: No encuentro el archivo .pt.")
            return

        print(f"Cargando grafo dinámico ({path_grafo})...")
        # weights_only=False para evitar el error de seguridad de PyTorch 2.6
        data = torch.load(path_grafo, weights_only=False)

        # Verificar que tenemos la lista dinámica
        if not hasattr(data, 'dynamic_edge_indices'):
            print("❌ Error: Tu objeto Data no tiene 'dynamic_edge_indices'.")
            print("Asegúrate de haber añadido la lógica del grafo dinámico.")
            return

        # 2. CALCULAR POSICIONES FIJAS (Layout Maestro)
        # Usamos el grafo estático (promedio) para decidir dónde se sienta cada alumno.
        # Si recalculáramos la posición cada semana, los nodos saltarían y marearía.
        print("Calculando distribución espacial de los alumnos...")
        G_static = to_networkx(data, to_undirected=True)
        
        # Layout fijo para toda la animación
        pos = nx.spring_layout(G_static, seed=42, k=0.3) 
        
        # Colores fijos (Nota Final)
        node_colors = data.y.numpy().flatten()
        
        # 3. GENERAR FRAMES SEMANA A SEMANA
        frames_dir = GRAPH_FRAMES_DIR
        if not clean_frames:
            frames_dir = frames_dir + name + '/'
        else:
            frames_dir = frames_dir + 'tmp_frames/'
        os.makedirs(frames_dir, exist_ok=True)

        # Borrar frames antiguos si existen
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        
        images = []
        num_weeks = len(data.dynamic_edge_indices)
        
        print(f"Generando {num_weeks} fotogramas...")
        
        for t in range(num_weeks):
            # Crear figura limpia
            plt.figure(figsize=(10, 8))
            
            # Construir el grafo DE ESTA SEMANA
            # Creamos un grafo vacío y añadimos solo las aristas del momento t
            G_t = nx.Graph()
            G_t.add_nodes_from(range(data.num_nodes))
            
            # Extraer aristas del tensor
            edge_index_t = data.dynamic_edge_indices[t]
            edges_t = edge_index_t.t().tolist()
            G_t.add_edges_from(edges_t)
            
            # DIBUJAR
            # 1. Nodos (en las posiciones FIJAS 'pos')
            nx.draw_networkx_nodes(G_t, pos, 
                                node_color=node_colors, 
                                cmap=plt.cm.RdYlGn, 
                                node_size=300, 
                                alpha=0.9)
            
            # 2. Aristas Dinámicas (Las que existen esta semana)
            nx.draw_networkx_edges(G_t, pos, alpha=0.4, edge_color='blue')
            
            # 3. Etiquetas
            labels = {i: str(i+1) for i in range(data.num_nodes)}
            nx.draw_networkx_labels(G_t, pos, labels, font_size=8)
            
            plt.title(f'Semana {t+1} - Dinámica Social de la Clase\nConexiones basadas en similitud semanal', fontsize=15)
            plt.axis('off')
            
            # Guardar frame
            filename = os.path.join(frames_dir, f'frame_{t}.png')
            plt.savefig(filename, dpi=100)
            plt.close() # Cerrar para liberar memoria
            
            images.append(imageio.imread(filename))
            print(f"  -> Frame {t+1}/{num_weeks} generado.")

        # 4. CREAR GIF
        output_gif = GRAPH_MEDIA_DIR + f'{name}.gif'
        # duration es el tiempo por frame en segundos (0.8s es un buen ritmo)
        imageio.mimsave(output_gif, images, duration=1000, loop=0) 
        
        print(f"✅ GIF GUARDADO EN: {os.path.abspath(output_gif)}")

        if clean_frames:
            # Limpieza (Borrar carpeta temporal de imágenes)
            for f in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, f))
            os.rmdir(frames_dir)
            print("🧹 Archivos temporales eliminados.")

    def visualize_all_graphs(self):
        """
        Visualiza todos los grafos en la carpeta GRAPH_DIR.
        """
        for file in os.listdir(GRAPH_DIR):
            if file.endswith('.pt'):
                path = os.path.join(GRAPH_DIR, file)
                name = file.replace('.pt', '')
                
                if 'dynamic' in file:
                    print(f"Visualizando grafo dinámico: {name}")
                    self.visualize_dynamic(path, name=name, clean_frames=True)
                else:
                    print(f"Visualizando grafo estático: {name}")
                    self.visualize_static(path, name=name, show=False)

if __name__ == "__main__":
    visualizer = GraphVisualizer()
    #visualizer.visualize_all_graphs()
    visualizer.visualize_dynamic(path_grafo=GRAPH_DIR + 'Temp_a&g_dynamic_graph_5NN.pt', name='grafo_dinamico', clean_frames=False)