"""
Módulo que define las arquitecturas de modelos GNN y LSTM para la predicción de rendimiento académico.
Contiene:
- My_LSTM: Modelo LSTM puro para modelar secuencias temporales sin información de grafo.
- GNN_GCN: Modelo basado en Graph Convolutional Networks para grafos estáticos
- GNN_GAT: Modelo basado en Graph Attention Networks para grafos estáticos.
- GNN_SAGE: Modelo basado en GraphSAGE para grafos estáticos.
- STGNN: Modelo espacio-temporal que combina GNN con LSTM para capturar dinámicas estructurales a lo largo del tiempo.
- AdaptiveModel: Clase envoltorio que permite seleccionar dinámicamente el tipo de modelo a utilizar en función de la configuración deseada.
"""
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Linear, Dropout, LSTM, ModuleList
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import sort_edge_index
    TORCH_DISPONIBLE = True
    print("✓ PyTorch y PyTorch Geometric detectados")
except ImportError:
    TORCH_DISPONIBLE = False
    print("⚠️  PyTorch Geometric no está instalado.")
    print("   Instalación: pip install torch torch-geometric")
    print("   El script mostrará solo la estructura de los modelos.")


# ============================================================================
# MODELOS GNN
# ============================================================================

if TORCH_DISPONIBLE:
    
    class My_LSTM(nn.Module):
        """
        Modelo LSTM puro para predicción basada exclusivamente en información temporal.

        Este modelo actúa como baseline, ya que únicamente modela la evolución
        temporal de las características de cada nodo (por ejemplo, estudiantes),
        ignorando cualquier estructura relacional o información del grafo.
        """

        def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.3, batch_first=True):
            """
            Inicializa el modelo LSTM.

            :param input_dim: Dimensión del vector de entrada por instante temporal.
            :param hidden_dim: Dimensión del estado oculto de la LSTM.
            :param num_layers: Número de capas apiladas en la LSTM.
            :param output_dim: Dimensión de la salida final.
            :param dropout: Probabilidad de dropout entre capas LSTM y antes de la capa final.
            :param batch_first: Si True, el tensor de entrada tiene forma [batch, seq, features].
            """
            super(My_LSTM, self).__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.sigmoid = nn.Sigmoid()

        def forward(self, data: Data):
            """
            Define el paso hacia adelante del modelo.

            :param data: Objeto Data de PyTorch Geometric que contiene:
                         - x: Tensor de forma [batch_size, seq_len, input_dim].
                         - edge_index: No utilizado (incluido por compatibilidad).
            :return: Tensor con la predicción final en rango [0,1].
            """
            lstm_out, (h_n, c_n) = self.lstm(data.x)
            last_hidden_state = lstm_out[:, -1, :] 
            out = self.dropout(last_hidden_state)
            out = self.fc(out)
            return self.sigmoid(out).squeeze()
        

    class GNN_GCN(torch.nn.Module):
        """
        Modelo Graph Convolutional Network (GCN) para predicción basada en grafos estáticos.

        Aplica múltiples capas de convolución espectral sobre el grafo,
        agregando información estructural de los vecinos de cada nodo.
        """

        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3):
            """
            Inicializa la arquitectura GCN.

            :param input_dim: Dimensión de las características de entrada.
            :param hidden_dim: Dimensión de las representaciones ocultas.
            :param output_dim: Dimensión de la salida final.
            :param num_layers: Número total de capas GCN.
            :param dropout: Probabilidad de dropout entre capas.
            """
            super(GNN_GCN, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
        
        def forward(self, data):
            """
            Ejecuta la propagación hacia adelante del modelo GCN.

            :param data: Objeto Data que contiene:
                         - x: Características nodales [N, F] o [N, T, F].
                         - edge_index: Índices de aristas del grafo.
            :return: Predicción escalar por nodo en rango [0,1].
            """
            x, edge_index = data.x, data.edge_index
            if x.dim() == 3:
                x = x.mean(dim=1)

            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc(x)
            return torch.sigmoid(x).squeeze()


    class GNN_GAT(torch.nn.Module):
        """
        Modelo Graph Attention Network (GAT).

        Utiliza mecanismos de atención para ponderar la contribución
        de los nodos vecinos durante la agregación.
        """

        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3, heads=4):
            """
            Inicializa la arquitectura GAT.

            :param input_dim: Dimensión de entrada.
            :param hidden_dim: Dimensión interna por cabeza de atención.
            :param output_dim: Dimensión de salida.
            :param num_layers: Número total de capas GAT.
            :param dropout: Probabilidad de dropout.
            :param heads: Número de cabezas de atención.
            """
            super(GNN_GAT, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, data):
            """
            Ejecuta la propagación hacia adelante del modelo GAT.

            :param data: Objeto Data con características nodales y aristas.
            :return: Predicción escalar por nodo en rango [0,1].
            """
            x, edge_index = data.x, data.edge_index
            if x.dim() == 3:
                x = x.mean(dim=1)

            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc(x)
            return torch.sigmoid(x).squeeze()


    class GNN_SAGE(torch.nn.Module):
        """
        Modelo GraphSAGE.

        Implementa un esquema de agregación inductiva que permite
        generalizar a nodos no vistos durante el entrenamiento.
        """

        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3, aggr="lstm"):
            """
            Inicializa la arquitectura GraphSAGE.

            :param input_dim: Dimensión de entrada.
            :param hidden_dim: Dimensión interna.
            :param output_dim: Dimensión de salida.
            :param num_layers: Número de capas.
            :param dropout: Probabilidad de dropout.
            :param aggr: Tipo de agregador (mean, max, lstm, etc.).
            """
            super(GNN_SAGE, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, data):
            """
            Ejecuta la propagación hacia adelante del modelo GraphSAGE.

            :param data: Objeto Data con x y edge_index.
            :return: Predicción escalar por nodo en rango [0,1].
            """
            x, edge_index = data.x, data.edge_index
            edge_index = sort_edge_index(edge_index, sort_by_row=False)

            if x.dim() == 3:
                x = x.mean(dim=1)

            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc(x)
            return torch.sigmoid(x).squeeze()


    class STGNN(torch.nn.Module):
        """
        Modelo Espacio-Temporal basado en GNN + LSTM.

        Combina procesamiento espacial (mediante capas de convolución
        sobre grafos) con modelado temporal secuencial (LSTM),
        permitiendo capturar dinámicas estructurales a lo largo del tiempo.
        """

        def __init__(self, type: str, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3):
            """
            Inicializa el modelo STGNN.

            :param type: Tipo de convolución espacial ('GCN', 'GAT', 'SAGE').
            :param input_dim: Dimensión de entrada.
            :param hidden_dim: Dimensión interna.
            :param output_dim: Dimensión de salida.
            :param num_layers: Número de capas espaciales.
            :param dropout: Probabilidad de dropout.
            """
            super(STGNN, self).__init__()
            
            self.num_layers = num_layers
            self.dropout_rate = dropout
            self.gcn_layers = ModuleList()
            
            HEADS = 2
            match type:
                case 'GCN':
                    self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
                case 'GAT':
                    self.gcn_layers.append(GATConv(input_dim, hidden_dim, heads=HEADS))
                case 'SAGE':
                    self.gcn_layers.append(SAGEConv(input_dim, hidden_dim, aggr="lstm"))
            
            for i in range(num_layers-2):
                if i == 0 and type == 'GAT':
                    self.gcn_layers.append(GCNConv(hidden_dim * HEADS, hidden_dim))
                else:
                    self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            
            if type == 'GAT' and num_layers <=2:
                self.lstm = LSTM(input_size=hidden_dim * HEADS, hidden_size=hidden_dim, batch_first=True)
            else:
                self.lstm = LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
            
            self.fc = Linear(hidden_dim, output_dim)

        def forward(self, data):
            """
            Ejecuta la propagación espacio-temporal.

            :param data: Objeto Data que contiene:
                         - x: [N, T, F]
                         - edge_index: Grafo estático
                         - dynamic_edge_indices: Lista opcional de grafos dinámicos por instante
            :return: Predicción escalar por nodo en rango [0,1].
            """
            x = data.x
            static_edge_index = data.edge_index
            batch_size, seq_len, _ = x.shape
            dyn_edges = getattr(data, "dynamic_edge_indices", None)
            
            embeddings_temporales = []
            
            for t in range(seq_len):
                x_t = x[:, t, :] 
                
                if dyn_edges is not None and t < len(dyn_edges):
                    current_edge_index = dyn_edges[t]
                else:
                    current_edge_index = static_edge_index
                
                current_edge_index = sort_edge_index(current_edge_index, sort_by_row=False)

                for layer in self.gcn_layers:
                    x_t = layer(x_t, current_edge_index)
                    x_t = F.relu(x_t)
                    x_t = F.dropout(x_t, p=self.dropout_rate, training=self.training)
                
                embeddings_temporales.append(x_t)
            
            x_sequence = torch.stack(embeddings_temporales, dim=1)
            lstm_out, _ = self.lstm(x_sequence)
            last_hidden_state = lstm_out[:, -1, :]
            out = self.fc(last_hidden_state)
            return torch.sigmoid(out).squeeze()
    

    class AdaptiveModel(nn.Module):
        """
        Modelo envoltorio que permite seleccionar dinámicamente
        el tipo de arquitectura a utilizar.
        """

        TYPES = ['LSTM', 'GCN', 'GAT', 'SAGE', 'STGNN']

        def __init__(self, model_type, input_dim, hidden_dim = 64, output_dim=1, num_layers=3, dropout=0.3, type_stgnn='GAT'):
            """
            Inicializa el modelo adaptativo.

            :param model_type: Tipo de modelo ('LSTM', 'GCN', 'GAT', 'SAGE', 'STGNN').
            :param input_dim: Dimensión de entrada.
            :param hidden_dim: Dimensión interna.
            :param output_dim: Dimensión de salida.
            :param num_layers: Número de capas.
            :param dropout: Probabilidad de dropout.
            :param type_stgnn: Tipo de capa espacial en caso de usar STGNN.
            """
            super().__init__()
            self.type = model_type
            
            match model_type:
                case 'LSTM':
                    self.model = My_LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)
                case 'GCN':
                    self.model = GNN_GCN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
                case 'GAT':
                    self.model = GNN_GAT(input_dim, hidden_dim, heads=2, num_layers=num_layers, dropout=dropout)
                case 'SAGE':
                    self.model = GNN_SAGE(input_dim, hidden_dim, aggr="lstm", num_layers=num_layers, dropout=dropout)
                case 'STGNN':
                    self.model = STGNN(type_stgnn, input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
                case _:
                    raise ValueError(f"Modelo desconocido: {model_type}")

            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, data):
            """
            Ejecuta la propagación hacia adelante delegando en el modelo seleccionado.

            :param data: Objeto Data con las entradas necesarias.
            :return: Predicción generada por el modelo subyacente.
            """
            return self.model.forward(data)