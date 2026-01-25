
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
import numpy as np

import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Linear, Dropout, LSTM, ModuleList
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data, Batch
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
        def __init__(self, model_type, input_dim, hidden_dim = 64, output_dim=1, num_layers=3, dropout=0.3):
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
                    self.model = STGNN('GAT', input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
                case _:
                    raise ValueError(f"Modelo desconocido: {model_type}")
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, data):
            return self.model.forward(data)
    
# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================
class EntrenadorGNN:
    """Entrenar y evaluar modelos GNN de forma robusta"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {self.device}\n")
    
    def shake_weights(self, model, std=0.01):
        """Añade ruido gaussiano a los pesos para escapar de mínimos locales"""
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * std
                param.add_(noise)
        print(f"🫨 SHAKE! Ruido inyectado a los pesos (std={std})")
    
    def entrenar(self, data, type: str, train_idx, test_idx, 
                          config=None):
        """
        Args:
            data: Objeto Data de PyG.
            type: Tipo de modelo (ej: 'GCN', 'LSTM', etc.).
            train_idx, test_idx: Tensores con los índices FIJOS.
            config: Diccionario con hiperparámetros (opcional).
        """
        # Configuración por defecto si no se pasa nada
        cfg = {
            'epochs': 500,
            'lr': 0.01,
            'hidden_dim': 32,  
            'dropout': 0.2,
            'paciencia': 50,
            'num_layers': 2,
            'max_restarts': 3
        }
        if config: cfg.update(config)

        # Mover datos al dispositivo
        data = data.to(self.device)
        train_idx = train_idx.to(self.device)
        test_idx = test_idx.to(self.device)

        # Instanciar Modelo Dinámicamente
        if data.x.dim() == 3:
            # Datos temporales [N, Semanas, Features]
            input_dim = data.x.size(2)
        else:
            # Datos estáticos [N, Features]
            input_dim = data.x.size(1)
            
        modelo = AdaptiveModel(
            model_type=type,
            input_dim=input_dim,
            hidden_dim=cfg['hidden_dim'], 
            output_dim=1,
            num_layers=cfg['num_layers'], 
            dropout=cfg['dropout']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(modelo.parameters(), lr=cfg['lr'], weight_decay=5e-4)
        criterion = torch.nn.MSELoss()

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer, mode='max', factor=0.5, patience=10
        #)
        
        # Variables de control
        mejor_metric_test = -float('inf') 
        contador_paciencia = 0
        restarts_done = 0
        mejor_modelo_state = None
        
        print(f"🚀 Iniciando training: {type} para {cfg['epochs']} epochs...")
        
        for epoch in range(cfg['epochs']):
            # --- TRAINING ---
            modelo.train()
            optimizer.zero_grad()
            
            out = modelo(data)
            
            # Calcular loss solo en nodos de entrenamiento
            loss = criterion(out[train_idx], data.y[train_idx].squeeze())
            
            loss.backward()
            optimizer.step()
            
            # --- EVALUACIÓN (Cada época para Early Stopping) ---
            modelo.eval()
            with torch.no_grad():
                pred = modelo(data)
                # Calculamos MAE en Test para monitorear generalización
                # (En TFG pequeño, usamos Test como Validación)
                y_pred_test = pred[test_idx].cpu().numpy().flatten() * 10 # Scale 0-10 if needed
                y_true_test = data.y[test_idx].cpu().numpy().flatten() * 10
                
                current_r2 = r2_score(y_true_test, y_pred_test)

                #scheduler.step(current_r2)
            # --- LOGGING & EARLY STOPPING ---
            if epoch % 10 == 0:
                test_mae = torch.mean(torch.abs(pred[test_idx] - data.y[test_idx])).item()
                train_mae = torch.mean(torch.abs(pred[train_idx] - data.y[train_idx])).item()
                print(f"Epoch {epoch:3d} | R2: {current_r2:.4f} | Train Loss: {loss.item():.4f} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")

            # Guardamos el modelo si mejora el TEST R2 (no el train loss)
            if current_r2 > mejor_metric_test:
                mejor_metric_test = current_r2
                contador_paciencia = 0
                # Guardamos copia de los pesos del mejor momento
                mejor_modelo_state = modelo.state_dict()
            else:
                contador_paciencia += 1
                if contador_paciencia >= cfg['paciencia']:
                    if restarts_done < cfg['max_restarts']:
                        print(f"🛑 Paciencia agotada en epoch {epoch}. Intentando revivir...")
                        
                        # 1. Cargar el mejor estado hasta ahora (para no agitar basura)
                        if mejor_modelo_state:
                            modelo.load_state_dict(mejor_modelo_state)
                        
                        # 2. Hacer Shake (añadir ruido)
                        self.shake_weights(modelo, std=0.08 * (0.8 ** restarts_done)) # Ruido decreciente
                        
                        # 3. Reiniciar optimizador con LR más bajo
                        current_lr = optimizer.param_groups[0]['lr']
                        new_lr = current_lr * 0.5
                        optimizer = torch.optim.Adam(modelo.parameters(), lr=new_lr, weight_decay=1e-4)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
                        
                        print(f"🔄 RESTART #{restarts_done+1}: LR bajado a {new_lr:.5f}, Paciencia reseteada.")

                        contador_paciencia = 0
                        restarts_done += 1
                    else:
                        print(f"❌ Máximos restarts alcanzados. Terminando entrenamiento en epoch {epoch} con R2 = {mejor_metric_test}.")
                        break
        
        # --- RESTAURAR MEJOR MODELO ---
        if mejor_modelo_state:
            modelo.load_state_dict(mejor_modelo_state)
        
        # --- REPORTE FINAL ---
        print("\n📊 Evaluación Final del Mejor Modelo.")
        modelo.eval()
        with torch.no_grad():
            pred = modelo(data)
            
            train_pred = pred[train_idx].cpu().numpy().flatten()
            train_true = data.y[train_idx].cpu().numpy().flatten()
            test_pred = pred[test_idx].cpu().numpy().flatten()
            test_true = data.y[test_idx].cpu().numpy().flatten()
            
            return self.calculate_metrics(test_true, test_pred, threshold=0.7), test_true, test_pred
        
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        """
        Calcula métricas completas de rendimiento para regresión y clasificación.
        
        Args:
            y_true: Tensor o array con notas reales (normalizadas 0-1).
            y_pred: Tensor o array con predicciones del modelo (normalizadas 0-1).
            threshold: Umbral de aprobado (0.5 si las notas son 0-1).
            
        Returns:
            Diccionario con todas las métricas.
        """
        # 1. Asegurar formato Numpy y desnormalizar para interpretación (0-10)
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
            
        # Aplanar arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Notas en escala 0-10 (para MAE/RMSE legibles)
        y_true_10 = y_true * 10
        y_pred_10 = y_pred * 10
        
        # --- MÉTRICAS DE REGRESIÓN ---
        mae = mean_absolute_error(y_true_10, y_pred_10)
        mse = mean_squared_error(y_true_10, y_pred_10)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_10, y_pred_10)
        
        # --- MÉTRICAS DE CLASIFICACIÓN (Pasar a Binario) ---
        # Asumimos: 1 = Aprueba, 0 = Suspende (O viceversa, ajusta según tu target)
        # Normalmente nota > 5 es aprobado.
        
        bin_true = (y_true >= threshold).astype(int)
        bin_pred = (y_pred >= threshold).astype(int)
        
        acc = accuracy_score(bin_true, bin_pred)
        # F1 Score centrado en la clase minoritaria (suponemos que es suspender o '0')
        # O 'weighted' si quieres media ponderada.
        f1 = f1_score(bin_true, bin_pred, average='weighted')
        
        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(bin_true, bin_pred).ravel()
        
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Accuracy": acc,
            "F1_Score": f1,
            "True_Negatives": tn, # Predijo suspenso, fue suspenso (Bien)
            "False_Negatives": fn, # Predijo suspenso, fue aprobado (Falsa Alarma)
            "False_Positives": fp, # Predijo aprobado, fue suspenso (PELIGRO)
            "True_Positives": tp   # Predijo aprobado, fue aprobado (Bien)
        }
        
        return metrics
            