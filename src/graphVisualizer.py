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
    """
    Clase encargada de generar representaciones visuales de los grafos
    académicos almacenados en formato PyTorch Geometric.

    Permite:
        - Visualización estática de grafos.
        - Visualización temporal mediante animaciones GIF para grafos dinámicos.
        - Procesamiento masivo de todos los grafos disponibles.
    """

    def visualize_static(self, path_grafo: str, week: int = None, name: str = 'grafo', show: bool = True):
        """
        Genera y guarda una visualización estática de un grafo almacenado en formato .pt.

        Si el grafo contiene estructura dinámica y se especifica una semana válida,
        se utilizará la estructura de aristas correspondiente a dicha semana.

        Args:
            path_grafo (str): Ruta al archivo .pt del grafo.
            week (int, optional): Semana específica en grafos dinámicos.
            name (str): Nombre base del archivo de salida.
            show (bool): Indica si se debe mostrar la figura en pantalla.
        """
        if not os.path.exists(path_grafo):
            print(f"Archivo no encontrado en la ruta especificada: {path_grafo}")
            print("Por favor, genere el grafo ejecutando primero el módulo correspondiente.")
            exit()

        print(f"Cargando el grafo desde: {path_grafo}")
        data = torch.load(path_grafo, weights_only=False)

        if week is not None and hasattr(data, 'dynamic_edge_indices') and week < len(data.dynamic_edge_indices):
            print(f"Aplicando la estructura de aristas correspondiente a la semana {week}.")
            data.edge_index = data.dynamic_edge_indices[week]

        G = to_networkx(data, to_undirected=True)

        plt.figure(figsize=(12, 8))

        node_colors = data.y.numpy().flatten()

        pos = nx.spring_layout(G, seed=42, k=0.3)

        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            cmap=plt.cm.RdYlGn,
            node_size=300,
            alpha=0.9
        )

        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

        labels = {i: str(i + 1) for i in range(data.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

        plt.colorbar(nodes, label='Nota Final Normalizada (0=0, 1=10)')
        plt.title('Grafo de Estudiantes (k-NN Similitud)\nColor = Rendimiento Académico', fontsize=15)
        plt.axis('off')

        output_file = GRAPH_MEDIA_DIR + f'{name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Imagen exportada en: {output_file}")

        if show:
            plt.show()

    def visualize_dynamic(self, path_grafo: str, name: str = 'grafo_dinamico', clean_frames: bool = True):
        """
        Genera una animación GIF a partir de un grafo dinámico almacenado en formato .pt.

        La posición espacial de los nodos se fija utilizando el layout del grafo
        estático para garantizar coherencia visual entre semanas.

        Args:
            path_grafo (str): Ruta al archivo .pt del grafo dinámico.
            name (str): Nombre base del archivo GIF resultante.
            clean_frames (bool): Indica si se deben eliminar los fotogramas
                                 intermedios tras generar el GIF.
        """
        if not os.path.exists(path_grafo):
            print("Error: no se encontró el archivo .pt especificado.")
            return

        print(f"Cargando grafo dinámico ({path_grafo})...")
        data = torch.load(path_grafo, weights_only=False)

        if not hasattr(data, 'dynamic_edge_indices'):
            print("El objeto Data proporcionado carece del atributo 'dynamic_edge_indices'.")
            print("Asegúrese de que el grafo dinámico haya sido construido correctamente.")
            return

        print("Calculando la distribución espacial de los nodos...")
        G_static = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G_static, seed=42, k=0.3)

        node_colors = data.y.numpy().flatten()

        frames_dir = GRAPH_FRAMES_DIR
        if not clean_frames:
            frames_dir = frames_dir + name + '/'
        else:
            frames_dir = frames_dir + 'tmp_frames/'
        os.makedirs(frames_dir, exist_ok=True)

        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))

        images = []
        num_weeks = len(data.dynamic_edge_indices)

        print(f"Generando {num_weeks} fotogramas de visualización...")

        for t in range(num_weeks):
            plt.figure(figsize=(10, 8))

            G_t = nx.Graph()
            G_t.add_nodes_from(range(data.num_nodes))

            edge_index_t = data.dynamic_edge_indices[t]
            edges_t = edge_index_t.t().tolist()
            G_t.add_edges_from(edges_t)

            nx.draw_networkx_nodes(
                G_t, pos,
                node_color=node_colors,
                cmap=plt.cm.RdYlGn,
                node_size=300,
                alpha=0.9
            )

            nx.draw_networkx_edges(G_t, pos, alpha=0.4, edge_color='blue')

            labels = {i: str(i + 1) for i in range(data.num_nodes)}
            nx.draw_networkx_labels(G_t, pos, labels, font_size=8)

            plt.title(
                f'Semana {t+1} - Dinámica Social de la Clase\nConexiones basadas en similitud semanal',
                fontsize=15
            )
            plt.axis('off')

            filename = os.path.join(frames_dir, f'frame_{t}.png')
            plt.savefig(filename, dpi=100)
            plt.close()

            images.append(imageio.imread(filename))
            print(f"  -> Fotograma {t+1}/{num_weeks} generado.")

        output_gif = GRAPH_MEDIA_DIR + f'{name}.gif'
        imageio.mimsave(output_gif, images, duration=1000, loop=0)

        print(f"GIF generado en: {os.path.abspath(output_gif)}")

        if clean_frames:
            for f in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, f))
            os.rmdir(frames_dir)
            print("Archivos temporales eliminados.")

    def visualize_all_graphs(self):
        """
        Genera visualizaciones para todos los grafos almacenados en GRAPH_DIR.

        - Los grafos estáticos se exportan como imágenes PNG.
        - Los grafos dinámicos se exportan como animaciones GIF.
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