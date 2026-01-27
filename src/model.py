
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
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.3, batch_first=True):
            """
            LSTM Pura para predicción de rendimiento.
            Actúa como BASELINE: Solo mira la evolución temporal, ignora las relaciones sociales.
            """
            super(My_LSTM, self).__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # DEFINICIÓN DE LA LSTM
            # batch_first=True es vital porque tus datos son [Batch, Semanas, Features]
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # CAPA COMPLETAMENTE CONECTADA (Decodificador)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.sigmoid = nn.Sigmoid()

        def forward(self, data: Data):
            """
            data: objeto Data de PyTorch Geometric
            x: Tensor [Batch_Size, Seq_Len, Input_Dim]
            (Ej: 70 alumnos, 15 semanas, 12 features)
            
            edge_index: SE IGNORA (Está aquí solo por compatibilidad con el trainer de GNN)
            """
            
            # 1. PASO LSTM
            # out: contiene los estados ocultos de TODAS las semanas [Batch, Seq, Hidden]
            # _ : (h_n, c_n) contiene el estado final (memoria)
            lstm_out, (h_n, c_n) = self.lstm(data.x)
            
            # 2. SELECCIÓN DEL ÚLTIMO ESTADO
            # Solo nos importa lo que el modelo "pensaba" en la última semana del curso.
            # Cogemos: Todas las filas (:), Última columna (-1), Todas las features (:)
            last_hidden_state = lstm_out[:, -1, :] 
            
            # 3. CLASIFICACIÓN / REGRESIÓN
            out = self.dropout(last_hidden_state)
            out = self.fc(out)
            
            # Retornamos sigmoid para tener rango 0-1 (nota normalizada)
            return self.sigmoid(out).squeeze()
        

    class GNN_GCN(torch.nn.Module):
        """Graph Convolutional Network"""
        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3):
            super(GNN_GCN, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
        
        def forward(self, data):
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
        """Graph Attention Network"""
        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3, heads=4):
            super(GNN_GAT, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, data):
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
        """GraphSAGE"""
        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3, aggr="lstm"):
            super(GNN_SAGE, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, data):
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
        """Modelo modular para grafos espaciales-temporales (Corregido)"""
        def __init__(self, type: str, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3):
            super(STGNN, self).__init__()
            
            self.num_layers = num_layers
            self.dropout_rate = dropout
            
            # --- FIX 1: Usar ModuleList para que PyTorch rastree los parámetros ---
            self.gcn_layers = ModuleList()
            
            # Primera capa (Input -> Hidden)
            HEADS = 2
            match type:
                case 'GCN':
                    self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
                case 'GAT':
                    self.gcn_layers.append(GATConv(input_dim, hidden_dim, heads=HEADS))
                case 'SAGE':
                    self.gcn_layers.append(SAGEConv(input_dim, hidden_dim, aggr="lstm"))
            
            # Capas ocultas siguientes (Hidden -> Hidden)
            for i in range(num_layers-2):
                if i == 0 and type == 'GAT':
                    self.gcn_layers.append(GCNConv(hidden_dim * HEADS, hidden_dim))
                else:
                    self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            
            # LSTM
            if type == 'GAT' and num_layers <=2:
                self.lstm = LSTM(input_size=hidden_dim * HEADS, hidden_size=hidden_dim, batch_first=True)
            else:
                self.lstm = LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
            
            # Capa de salida
            self.fc = Linear(hidden_dim, output_dim)

        def forward(self, data):
            """
            data: Objeto Data de PyG que contiene:
                - x: [N_alumnos, N_semanas, N_features]
                - edge_index: Grafo estático (fallback)
                - dynamic_edge_indices: Lista de aristas por semana (opcional)
            """
            x = data.x
            static_edge_index = data.edge_index
            batch_size, seq_len, _ = x.shape

            # Comprobamos si existen aristas dinámicas
            dyn_edges = getattr(data, "dynamic_edge_indices", None)
            
            embeddings_temporales = []
            
            # --- BUCLE TEMPORAL (Semana a Semana) ---
            for t in range(seq_len):
                # 1. Extraer features de la semana t: [N, Features]
                x_t = x[:, t, :] 
                
                # 2. Decidir qué grafo usar (Dinámico o Estático)
                # --- FIX 2: Lógica robusta ---
                if dyn_edges is not None and t < len(dyn_edges):
                    current_edge_index = dyn_edges[t]
                else:
                    current_edge_index = static_edge_index # Usamos el estático por defecto
                
                current_edge_index = sort_edge_index(current_edge_index, sort_by_row=False)
                # 3. Aplicar capas GCN
                for layer in self.gcn_layers:
                    x_t = layer(x_t, current_edge_index)
                    x_t = F.relu(x_t)
                    x_t = F.dropout(x_t, p=self.dropout_rate, training=self.training)
                
                # Guardamos el embedding espacial de esta semana
                embeddings_temporales.append(x_t)
            
            # 4. Reconstruir secuencia: [N, Semanas, Hidden]
            x_sequence = torch.stack(embeddings_temporales, dim=1)
            
            # 5. LSTM (Procesamiento Temporal)
            # Solo necesitamos el output, (h_n, c_n) no se usan explícitamente si cogemos out[:, -1, :]
            lstm_out, _ = self.lstm(x_sequence)
            
            # Tomamos el estado de la ÚLTIMA semana para predecir
            last_hidden_state = lstm_out[:, -1, :]
            
            # 6. Predicción Final
            out = self.fc(last_hidden_state)
            
            # Sigmoid asume que tus notas target están normalizadas 0-1
            return torch.sigmoid(out).squeeze()
    

    class AdaptiveModel(nn.Module):
        TYPES = ['LSTM', 'GCN', 'GAT', 'SAGE', 'STGNN']
        def __init__(self, model_type, input_dim, hidden_dim = 64, output_dim=1, num_layers=3, dropout=0.3, type_stgnn='GAT'):
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
            return self.model.forward(data)
    
