"""
PREDICCIÓN DE RENDIMIENTO ACADÉMICO CON GRAPH NEURAL NETWORKS (GNN)

Implementa 3 enfoques diferentes:
1. GRAFO ESTÁTICO DE ESTUDIANTES: Nodos=Estudiantes, Aristas=Similitud
2. GRAFO TEMPORAL DINÁMICO: Serie de grafos (uno por semana)
3. GRAFO HETEROGÉNEO: Nodos=Estudiantes+Semanas (bipartito)

Requiere: pip install torch torch-geometric scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Intentar importar PyTorch y PyTorch Geometric
try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear, Dropout
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_DISPONIBLE = True
    print("✓ PyTorch y PyTorch Geometric detectados")
except ImportError:
    TORCH_DISPONIBLE = False
    print("⚠️  PyTorch Geometric no está instalado.")
    print("   Instalación: pip install torch torch-geometric")
    print("   El script mostrará solo la estructura de los modelos.")


class CargadorDatos:
    """Cargar y preparar datos para GNN"""
    
    def __init__(self, ruta_datos='./'):
        self.ruta_datos = ruta_datos
        
    def cargar_todos(self):
        """Cargar todos los datos"""
        print("\n📂 Cargando datos...")
        
        # Archivos principales
        self.asistencia = pd.read_csv(f'{self.ruta_datos}asistencia.csv')
        self.seguimiento = pd.read_csv(f'{self.ruta_datos}seguimiento.csv')
        self.predecir = pd.read_csv(f'{self.ruta_datos}predecir.csv')
        self.predecir['final'] = self.predecir['final'].str.replace(',', '.').astype(float)
        
        # Cargar encuestas
        self.encuestas = []
        nombres_encuestas = [
            'encuestas/1.primera_03_03.csv', 
            'encuestas/2.segunda_10_03.csv', 
            'encuestas/3.tercera_17_03.csv',
            'encuestas/4.cuarta_24_03.csv', 
            'encuestas/5.quinta_31_03.csv', 
            'encuestas/6.sexta_07_04.csv',
            'encuestas/7.septima_21_04.csv', 
            'encuestas/8.octava_28_04 (APAGÓN).csv', 
            'encuestas/9.novena_05_05.csv',
            'encuestas/10.décima_12_05.csv', 
            'encuestas/11.undécima_19_05.csv'
        ]
        
        for i, nombre in enumerate(nombres_encuestas, 1):
            try:
                df = pd.read_csv(f'{self.ruta_datos}{nombre}')
                df['semana'] = i
                self.encuestas.append(df)
                print(f"  ✓ Encuesta semana {i}: {len(df)} respuestas")
            except FileNotFoundError:
                print(f"  ⚠ Encuesta semana {i} no encontrada")
        
        print(f"✓ Datos cargados: {len(self.predecir)} estudiantes, {len(self.encuestas)} semanas\n")
        return self


# ============================================================================
# ENFOQUE 1: GRAFO ESTÁTICO DE ESTUDIANTES
# Nodos = Estudiantes | Aristas = Similitud en comportamiento
# ============================================================================

class GrafoEstaticoEstudiantes:
    """
    ENFOQUE 1: Grafo estático donde cada nodo es un estudiante
    
    Estructura:
    - Nodos: Estudiantes (73 nodos)
    - Features de nodo: Agregados temporales (media encuestas, asistencia, etc.)
    - Aristas: Similitud en respuestas de encuestas + grupos de prácticas
    - Objetivo: Aprovechar homofilia (estudiantes similares tienen rendimiento similar)
    """
    
    def __init__(self, cargador):
        self.cargador = cargador
        
    def crear_features_nodo(self):
        """Crear features agregadas por estudiante"""
        print("🔧 Creando features de nodos (estudiantes)...")
        
        features_por_estudiante = []
        estudiantes_ids = []
        
        for id_estudiante in self.cargador.predecir['ID'].values:
            estudiantes_ids.append(id_estudiante)
            
            # Agregar respuestas de encuestas
            respuestas = []
            for encuesta in self.cargador.encuestas:
                resp = encuesta[encuesta['ID'] == id_estudiante]
                if len(resp) > 0:
                    respuestas.append(resp.iloc[0])
            
            features = []
            
            # Features de encuestas
            if len(respuestas) > 0:
                df_resp = pd.DataFrame(respuestas)
                
                # Identificar columnas numéricas
                cols_num = [col for col in df_resp.columns 
                            if col not in ['ID', 'semana'] and 
                            df_resp[col].dtype in ['int64', 'float64']]
                
                # Media y std de cada pregunta
                for col in cols_num[:6]: # Primeras 6 preguntas
                    if len(cols_num) > 0:
                        features.append(df_resp[col].mean())
                        features.append(df_resp[col].std()) # std() da NaN si hay 1 muestra
                    else:
                        features.extend([0, 0])
                
                # Tendencias (primera mitad vs segunda mitad)
                if len(respuestas) >= 4:
                    mitad = len(respuestas) // 2
                    for col in cols_num[:3]:
                        primera = df_resp.iloc[:mitad][col].mean()
                        segunda = df_resp.iloc[mitad:][col].mean()
                        features.append(segunda - primera)
                else:
                    features.extend([0, 0, 0])
                
                # Tasa de respuesta
                features.append(len(respuestas) / len(self.cargador.encuestas))
            else:
                features.extend([0] * 16) # 6*2 + 3 + 1
            
            # Features de asistencia
            fila_asist = self.cargador.asistencia[
                self.cargador.asistencia['ID'] == id_estudiante
            ]
            if len(fila_asist) > 0:
                cols_asist = [col for col in self.cargador.asistencia.columns if col != 'ID']
                valores = fila_asist[cols_asist].values[0]
                
                total = len([v for v in valores if str(v) not in ['semana_santa', 'nan']])
                ausencias = sum(1 for v in valores if str(v) == 'NO')
                
                features.append(1 - (ausencias / total) if total > 0 else 0) # % asistencia
                features.append(ausencias) # total ausencias
            else:
                features.extend([0, 0])
            
            # Features de prácticas
            fila_prac = self.cargador.seguimiento[
                self.cargador.seguimiento['ID'] == id_estudiante
            ]
            if len(fila_prac) > 0:
                cols_prac = [col for col in self.cargador.seguimiento.columns 
                             if 'P' in col and 'examen' not in col.lower()]
                
                notas = []
                for col in cols_prac:
                    try:
                        nota = float(str(fila_prac[col].values[0]).replace(',', '.'))
                        notas.append(nota)
                    except:
                        pass
                
                if len(notas) > 0:
                    features.append(np.mean(notas))
                    features.append(np.std(notas))
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            features_por_estudiante.append(features)
        
        features_array = np.array(features_por_estudiante, dtype=np.float32)
        
        # --- CORRECCIÓN NaN ---
        # Reemplazamos cualquier NaN (p.ej., de std() con 1 muestra) por 0.
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        # --- FIN CORRECCIÓN ---

        print(f"✓ Features creadas: {features_array.shape[0]} nodos, {features_array.shape[1]} features/nodo\n")
        
        return features_array, estudiantes_ids
    
    def crear_matriz_similitud(self, features_array):
        """Crear matriz de similitud entre estudiantes"""
        print("🔗 Creando aristas basadas en similitud...")
        
        # Similitud coseno entre features de estudiantes
        similitud = cosine_similarity(features_array)
        
        # Crear aristas para estudiantes con similitud > threshold
        threshold = 0.5 # Ajustable
        edge_index = []
        edge_weight = []
        
        for i in range(len(similitud)):
            for j in range(i+1, len(similitud)):
                if similitud[i, j] > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i]) # Grafo no dirigido
                    edge_weight.append(similitud[i, j])
                    edge_weight.append(similitud[i, j])
        
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_weight = np.array(edge_weight, dtype=np.float32)
        
        num_aristas = edge_index.shape[1]
        print(f"✓ Aristas creadas: {num_aristas} conexiones (threshold={threshold})\n")
        
        return edge_index, edge_weight
    
    def crear_grafo(self):
        """Crear objeto de grafo PyG"""
        features, ids = self.crear_features_nodo()
        edge_index, edge_weight = self.crear_matriz_similitud(features)
        
        # Normalizar features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        # Targets
        targets = self.cargador.predecir['final'].values
        
        if not TORCH_DISPONIBLE:
            print("⚠️ PyTorch no disponible, retornando datos numpy")
            return features_norm, edge_index, targets, scaler
        
        # Crear objeto Data de PyTorch Geometric
        data = Data(
            x=torch.FloatTensor(features_norm),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_weight),
            y=torch.FloatTensor(targets)
        )
        
        print(f"📊 Grafo estático creado:")
        print(f"   - Nodos: {data.x.shape[0]}")
        print(f"   - Features/nodo: {data.x.shape[1]}")
        print(f"   - Aristas: {data.edge_index.shape[1]}")
        print()
        
        return data, scaler


# ============================================================================
# ENFOQUE 2: GRAFO TEMPORAL DINÁMICO
# Serie de grafos G1, G2, ..., G11 (uno por semana)
# ============================================================================

class GrafoTemporalDinamico:
    """
    ENFOQUE 2: Grafo temporal dinámico
    
    Estructura:
    - 11 grafos (uno por semana)
    - Cada grafo: Nodos = Estudiantes
    - Features: Respuestas de encuesta + asistencia de ESA semana
    - Aristas: Similitud en respuestas de ESA semana
    - Objetivo: Capturar evolución temporal de relaciones
    """
    
    def __init__(self, cargador):
        self.cargador = cargador
        
    def crear_grafo_semana(self, semana_idx):
        """Crear grafo para una semana específica"""
        encuesta = self.cargador.encuestas[semana_idx]
        
        # Estudiantes que respondieron esa semana
        estudiantes_semana = encuesta['ID'].values
        features_semana = []
        
        # Crear mapeo ID -> índice
        id_to_idx = {id_est: idx for idx, id_est in enumerate(estudiantes_semana)}
        
        # Features: respuestas de esa semana
        for id_est in estudiantes_semana:
            resp = encuesta[encuesta['ID'] == id_est].iloc[0]
            
            # Extraer respuestas numéricas
            cols_num = [col for col in encuesta.columns 
                        if col not in ['ID', 'semana'] and 
                        encuesta[col].dtype in ['int64', 'float64']]
            
            features = []
            for col in cols_num:
                try:
                    features.append(float(resp[col]))
                except:
                    features.append(0.0)
            
            # Agregar asistencia de esa semana
            fila_asist = self.cargador.asistencia[
                self.cargador.asistencia['ID'] == id_est
            ]
            if len(fila_asist) > 0:
                cols_asist = [col for col in self.cargador.asistencia.columns if col != 'ID']
                if semana_idx < len(cols_asist):
                    valor = str(fila_asist[cols_asist[semana_idx]].values[0])
                    features.append(1.0 if valor == '' else 0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            features_semana.append(features)
        
        features_array = np.array(features_semana, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Crear aristas por similitud en respuestas
        similitud = cosine_similarity(features_array)
        threshold = 0.6
        
        edge_index = []
        for i in range(len(similitud)):
            for j in range(i+1, len(similitud)):
                if similitud[i, j] > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        
        if len(edge_index) == 0:
            # Grafo vacío, crear al menos algunas aristas
            for i in range(min(5, len(features_array))):
                for j in range(i+1, min(5, len(features_array))):
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        
        edge_index = np.array(edge_index, dtype=np.int64).T if len(edge_index) > 0 else np.array([[], []], dtype=np.int64)
        
        # Targets para estos estudiantes
        targets = []
        for id_est in estudiantes_semana:
            target = self.cargador.predecir[
                self.cargador.predecir['ID'] == id_est
            ]['final'].values[0]
            targets.append(target)
        
        targets = np.array(targets, dtype=np.float32)
        
        return features_array, edge_index, targets, estudiantes_semana
    
    def crear_grafos_temporales(self):
        """Crear serie de grafos temporales"""
        print("🔧 Creando grafos temporales (uno por semana)...")
        
        grafos = []
        
        for sem_idx in range(len(self.cargador.encuestas)):
            features, edge_index, targets, ids = self.crear_grafo_semana(sem_idx)
            
            print(f"  Semana {sem_idx+1}: {len(ids)} nodos, {edge_index.shape[1]} aristas")
            
            if not TORCH_DISPONIBLE:
                grafos.append({
                    'features': features,
                    'edge_index': edge_index,
                    'targets': targets,
                    'ids': ids,
                    'semana': sem_idx + 1
                })
            else:
                # Normalizar features
                scaler = StandardScaler()
                features_norm = scaler.fit_transform(features)
                
                data = Data(
                    x=torch.FloatTensor(features_norm),
                    edge_index=torch.LongTensor(edge_index) if edge_index.shape[1] > 0 else torch.LongTensor([[], []]),
                    y=torch.FloatTensor(targets)
                )
                grafos.append(data)
        
        print(f"✓ {len(grafos)} grafos temporales creados\n")
        return grafos


# ============================================================================
# ENFOQUE 3: GRAFO HETEROGÉNEO BIPARTITO
# Nodos tipo 1 = Estudiantes | Nodos tipo 2 = Semanas
# ============================================================================

class GrafoHeterogeneo:
    """
    ENFOQUE 3: Grafo heterogéneo bipartito
    
    Estructura:
    - Nodos tipo 1: Estudiantes (73 nodos)
    - Nodos tipo 2: Semanas (11 nodos)
    - Aristas: Estudiante participó en semana X
    - Features nodo estudiante: Agregados temporales
    - Features nodo semana: Estadísticas de esa semana
    - Objetivo: Modelar relación estudiante-tiempo explícitamente
    """
    
    def __init__(self, cargador):
        self.cargador = cargador
        self.num_estudiantes = len(self.cargador.predecir)
        self.num_semanas = len(self.cargador.encuestas)
        
    def crear_features_estudiantes(self):
        """Features agregadas de estudiantes"""
        # Reutilizar del enfoque 1
        grafo_est = GrafoEstaticoEstudiantes(self.cargador)
        features, ids = grafo_est.crear_features_nodo()
        return features, ids
    
    def crear_features_semanas(self):
        """Features agregadas de cada semana"""
        print("🔧 Creando features de semanas...")
        
        features_semanas = []
        
        for encuesta in self.cargador.encuestas:
            # Estadísticas agregadas de esa semana
            cols_num = [col for col in encuesta.columns 
                        if col not in ['ID', 'semana'] and 
                        encuesta[col].dtype in ['int64', 'float64']]
            
            features = []
            for col in cols_num[:6]:
                if len(cols_num) > 0:
                    features.append(encuesta[col].mean())
                    features.append(encuesta[col].std())
                else:
                    features.extend([0, 0])
            
            # Número de respuestas
            features.append(len(encuesta))
            
            # Tasa de respuesta
            features.append(len(encuesta) / self.num_estudiantes)
            
            features_semanas.append(features)
        
        features_array = np.array(features_semanas, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Features de semanas: {features_array.shape}\n")
        
        return features_array
    
    def crear_grafo_heterogeneo(self):
        """Crear grafo bipartito estudiante-semana"""
        print("🔧 Creando grafo heterogéneo bipartito...")
        
        # Features de estudiantes y semanas
        features_est, ids_est = self.crear_features_estudiantes()
        features_sem = self.crear_features_semanas()
        
        # Combinar features (primero estudiantes, luego semanas)
        # Padding para igualar dimensiones
        max_dim = max(features_est.shape[1], features_sem.shape[1])
        
        features_est_pad = np.pad(
            features_est, 
            ((0, 0), (0, max_dim - features_est.shape[1])), 
            mode='constant'
        )
        features_sem_pad = np.pad(
            features_sem,
            ((0, 0), (0, max_dim - features_sem.shape[1])),
            mode='constant'
        )
        
        features_all = np.vstack([features_est_pad, features_sem_pad])
        
        # Crear aristas estudiante -> semana
        edge_index = []
        
        for sem_idx, encuesta in enumerate(self.cargador.encuestas):
            semana_node_idx = self.num_estudiantes + sem_idx
            
            for id_est in encuesta['ID'].values:
                est_idx = np.where(ids_est == id_est)[0]
                if len(est_idx) > 0:
                    est_idx = est_idx[0]
                    # Arista bidireccional
                    edge_index.append([est_idx, semana_node_idx])
                    edge_index.append([semana_node_idx, est_idx])
        
        edge_index = np.array(edge_index, dtype=np.int64).T
        
        # Targets (solo para nodos de estudiantes)
        targets = self.cargador.predecir['final'].values
        targets_all = np.concatenate([targets, np.zeros(self.num_semanas)])
        
        # Máscara de nodos de estudiantes (solo estos tienen target)
        train_mask = np.concatenate([
            np.ones(self.num_estudiantes, dtype=bool),
            np.zeros(self.num_semanas, dtype=bool)
        ])
        
        print(f"✓ Grafo heterogéneo creado:")
        print(f"   - Nodos estudiantes: {self.num_estudiantes}")
        print(f"   - Nodos semanas: {self.num_semanas}")
        print(f"   - Total nodos: {len(features_all)}")
        print(f"   - Aristas: {edge_index.shape[1]}")
        print()
        
        if not TORCH_DISPONIBLE:
            return features_all, edge_index, targets_all, train_mask
        
        # Normalizar
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features_all)
        
        data = Data(
            x=torch.FloatTensor(features_norm),
            edge_index=torch.LongTensor(edge_index),
            y=torch.FloatTensor(targets_all),
            train_mask=torch.BoolTensor(train_mask) # Usaremos esta máscara para filtrar nodos
        )
        
        return data, scaler


# ============================================================================
# MODELOS GNN
# ============================================================================

if TORCH_DISPONIBLE:
    
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
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc(x)
            return x.squeeze()
    
    
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
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc(x)
            return x.squeeze()
    
    
    class GNN_SAGE(torch.nn.Module):
        """GraphSAGE"""
        def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, dropout=0.3):
            super(GNN_SAGE, self).__init__()
            
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc(x)
            return x.squeeze()
    
    
    class GNN_Temporal(torch.nn.Module):
        """GNN para grafos temporales (procesa secuencia de grafos)"""
        def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.3):
            super(GNN_Temporal, self).__init__()
            
            # GNN para procesar cada grafo
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            
            # LSTM para procesar secuencia temporal
            self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            
            self.fc = Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, grafos_lista):
            """
            grafos_lista: Lista de Data objects (uno por timestep)
            """
            embeddings_temporales = []
            
            for data in grafos_lista:
                x, edge_index = data.x, data.edge_index
                
                # Procesar grafo con GNN
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                
                # Pooling global (representación del grafo)
                x_graph = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
                embeddings_temporales.append(x_graph)
            
            # Stack embeddings temporales
            x_temporal = torch.stack(embeddings_temporales, dim=1) # [batch, time, hidden]
            
            # LSTM para capturar evolución temporal
            lstm_out, _ = self.lstm(x_temporal)
            
            # Usar último estado
            x_final = lstm_out[:, -1, :]
            
            # Predicción
            out = self.fc(x_final)
            return out.squeeze()


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

class EntrenadorGNN:
    """Entrenar y evaluar modelos GNN"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {self.device}\n")
    
    def entrenar_estatico(self, data, modelo_class, epochs=200, lr=0.01, use_mask=False):
        """
        Entrenar modelo en grafo estático.
        Si use_mask es True, usa data.train_mask para filtrar nodos (Enfoque 3).
        """
        
        # Split train/test
        if use_mask:
            # Para el grafo heterogéneo, los nodos a predecir están en la máscara
            node_indices = torch.where(data.train_mask)[0]
            num_nodes = len(node_indices)
        else:
            # Para el grafo estático, todos los nodos son estudiantes
            num_nodes = data.x.size(0)
            node_indices = torch.arange(num_nodes)

        indices_perm = torch.randperm(num_nodes)
        train_size = int(0.8 * num_nodes)
        
        # Índices dentro del subconjunto de nodos (p.ej., 0 a 72)
        train_sub_idx = indices_perm[:train_size]
        test_sub_idx = indices_perm[train_size:]
        
        # Mapear de vuelta a los índices globales del grafo
        train_idx = node_indices[train_sub_idx]
        test_idx = node_indices[test_sub_idx]

        # Modelo
        modelo = modelo_class(
            input_dim=data.x.size(1),
            hidden_dim=128, # Aumentado para más capacidad
            output_dim=1,
            num_layers=3,
            dropout=0.5 # Aumentado para regularizar
        ).to(self.device)
        
        optimizer = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=5e-4)
        criterion = torch.nn.MSELoss()
        
        data = data.to(self.device)
        
        # Entrenamiento
        mejor_loss = float('inf')
        paciencia = 50 # Aumentado
        contador = 0
        
        print(f"Iniciando entrenamiento para {epochs} epochs...")
        
        for epoch in range(epochs):
            modelo.train()
            optimizer.zero_grad()
            
            out = modelo(data)
            
            # Calcular loss solo en nodos de entrenamiento
            loss = criterion(out[train_idx], data.y[train_idx])
            
            loss.backward()
            optimizer.step()
            
            # Evaluación
            if epoch % 20 == 0 or epoch == epochs - 1: # Imprimir cada 20 epochs
                modelo.eval()
                with torch.no_grad():
                    pred = modelo(data)
                    
                    # Calcular MAE solo en nodos relevantes (train y test)
                    train_mae = torch.abs(pred[train_idx] - data.y[train_idx]).mean().item()
                    test_mae = torch.abs(pred[test_idx] - data.y[test_idx]).mean().item()
                    
                    print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.4f} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
                    
                    # Early stopping basado en loss de entrenamiento
                    if loss.item() < mejor_loss:
                        mejor_loss = loss.item()
                        contador = 0
                    else:
                        contador += 1
                    
                    if contador >= paciencia:
                        print(f"--- Early stopping en epoch {epoch} ---")
                        break
        
        # Evaluación final
        print("Evaluación final...")
        modelo.eval()
        with torch.no_grad():
            pred = modelo(data)
            
            train_pred = pred[train_idx].cpu().numpy()
            train_true = data.y[train_idx].cpu().numpy()
            test_pred = pred[test_idx].cpu().numpy()
            test_true = data.y[test_idx].cpu().numpy()
            
            # Evitar R² negativo por varianza cero (si solo hay 1 muestra de test)
            if len(test_true) < 2:
                test_r2 = 0.0
            else:
                 test_r2 = r2_score(test_true, test_pred)

            if len(train_true) < 2:
                train_r2 = 0.0
            else:
                 train_r2 = r2_score(train_true, train_pred)

            resultados = {
                'train_mae': mean_absolute_error(train_true, train_pred),
                'train_rmse': np.sqrt(mean_squared_error(train_true, train_pred)),
                'train_r2': train_r2,
                'test_mae': mean_absolute_error(test_true, test_pred),
                'test_rmse': np.sqrt(mean_squared_error(test_true, test_pred)),
                'test_r2': test_r2,
                'modelo': modelo
            }
            
            return resultados


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    print("="*80)
    print("  PREDICCIÓN DE RENDIMIENTO CON GRAPH NEURAL NETWORKS")
    print("="*80)
    
    if not TORCH_DISPONIBLE:
        print("\n⚠️  PyTorch Geometric no disponible.")
        print("   Instalación:")
        print("   pip install torch torch-geometric")
        print("\n   Mostrando solo estructura de grafos...\n")
    
    # Cargar datos
    cargador = CargadorDatos('./').cargar_todos()
    
    # --- Parámetros de Entrenamiento ---
    EPOCHS = 500 # Aumentado
    LR = 0.005 # Reducido para entrenamiento más estable 
    # --- Fin Parámetros ---

    print("\n" + "="*80)
    print("  ENFOQUE 1: GRAFO ESTÁTICO DE ESTUDIANTES")
    print("="*80)
    grafo1 = GrafoEstaticoEstudiantes(cargador)
    data1, scaler1 = grafo1.crear_grafo()
    
    if TORCH_DISPONIBLE:
        entrenador = EntrenadorGNN()
        
        print("🤖 Entrenando GCN en grafo estático...")
        resultados_gcn = entrenador.entrenar_estatico(
            data1, GNN_GCN, epochs=EPOCHS, lr=LR, use_mask=False
        )
        print(f"\n📊 Resultados GCN (Estático):")
        print(f"   Train - MAE: {resultados_gcn['train_mae']:.4f} | RMSE: {resultados_gcn['train_rmse']:.4f} | R²: {resultados_gcn['train_r2']:.4f}")
        print(f"   Test  - MAE: {resultados_gcn['test_mae']:.4f} | RMSE: {resultados_gcn['test_rmse']:.4f} | R²: {resultados_gcn['test_r2']:.4f}")
        
        print("\n🤖 Entrenando GAT en grafo estático...")
        resultados_gat = entrenador.entrenar_estatico(
            data1, GNN_GAT, epochs=EPOCHS, lr=LR, use_mask=False
        )
        print(f"\n📊 Resultados GAT (Estático):")
        print(f"   Train - MAE: {resultados_gat['train_mae']:.4f} | RMSE: {resultados_gat['train_rmse']:.4f} | R²: {resultados_gat['train_r2']:.4f}")
        print(f"   Test  - MAE: {resultados_gat['test_mae']:.4f} | RMSE: {resultados_gat['test_rmse']:.4f} | R²: {resultados_gat['test_r2']:.4f}")

        print("\n🤖 Entrenando GraphSAGE en grafo estático...")
        resultados_sage = entrenador.entrenar_estatico(
            data1, GNN_SAGE, epochs=EPOCHS, lr=LR, use_mask=False
        )
        print(f"\n📊 Resultados SAGE (Estático):")
        print(f"   Train - MAE: {resultados_sage['train_mae']:.4f} | RMSE: {resultados_sage['train_rmse']:.4f} | R²: {resultados_sage['train_r2']:.4f}")
        print(f"   Test  - MAE: {resultados_sage['test_mae']:.4f} | RMSE: {resultados_sage['test_rmse']:.4f} | R²: {resultados_sage['test_r2']:.4f}")

    
    print("\n" + "="*80)
    print("  ENFOQUE 2: GRAFO TEMPORAL DINÁMICO")
    print("="*80)
    grafo2 = GrafoTemporalDinamico(cargador)
    grafos_temporales = grafo2.crear_grafos_temporales()
    print("  Nota: El entrenamiento para GNN_Temporal (Enfoque 2) requiere")
    print("         una lógica de batching y entrenamiento temporal")
    print("         que no está implementada en 'entrenar_estatico'.")
    
    print("\n" + "="*80)
    print("  ENFOQUE 3: GRAFO HETEROGÉNEO BIPARTITO")
    print("="*80)
    grafo3 = GrafoHeterogeneo(cargador)
    
    if TORCH_DISPONIBLE:
        data3, scaler3 = grafo3.crear_grafo_heterogeneo()
        print("🤖 Entrenando GCN en grafo heterogéneo...")
        resultados_hetero_gcn = entrenador.entrenar_estatico(
            data3, GNN_GCN, epochs=EPOCHS, lr=LR, use_mask=True
        )
        print(f"\n📊 Resultados GCN (Heterogéneo):")
        print(f"   Train - MAE: {resultados_hetero_gcn['train_mae']:.4f} | RMSE: {resultados_hetero_gcn['train_rmse']:.4f} | R²: {resultados_hetero_gcn['train_r2']:.4f}")
        print(f"   Test  - MAE: {resultados_hetero_gcn['test_mae']:.4f} | RMSE: {resultados_hetero_gcn['test_rmse']:.4f} | R²: {resultados_hetero_gcn['test_r2']:.4f}")
        
        print("\n🤖 Entrenando GAT en grafo heterogéneo...")
        resultados_hetero_gat = entrenador.entrenar_estatico(
            data3, GNN_GAT, epochs=EPOCHS, lr=LR, use_mask=True
        )
        print(f"\n📊 Resultados GAT (Heterogéneo):")
        print(f"   Train - MAE: {resultados_hetero_gat['train_mae']:.4f} | RMSE: {resultados_hetero_gat['train_rmse']:.4f} | R²: {resultados_hetero_gat['train_r2']:.4f}")
        print(f"   Test  - MAE: {resultados_hetero_gat['test_mae']:.4f} | RMSE: {resultados_hetero_gat['test_rmse']:.4f} | R²: {resultados_hetero_gat['test_r2']:.4f}")

    else:
        # Si torch no está, solo muestra la creación del grafo
        data3, edge_index, targets, mask = grafo3.crear_grafo_heterogeneo()
    
    print("\n" + "="*80)
    print("  RESUMEN")
    print("="*80)
    print("""
    ENFOQUE 1 (GRAFO ESTÁTICO):
    ✓ Más simple y efectivo para un baseline
    ✓ Captura similitud entre estudiantes
    ✓ Funciona bien con GCN, GAT y SAGE
    
    ENFOQUE 2 (TEMPORAL DINÁMICO):
    ✓ Captura evolución temporal
    ✓ Requiere arquitectura más compleja (GNN + LSTM)
    ✓ Mejor para predicción temprana (si se entrena para ello)
    
    ENFOQUE 3 (HETEROGÉNEO):
    ✓ Modela relación estudiante-tiempo explícitamente
    ✓ Permite a los nodos "semana" actuar como agregadores
    ✓ Puede requerir HeteroGNN especializado para mejor rendimiento
    """)
    
    print("="*80)
    print("✅ Pipeline completado!")
    print("="*80)


if __name__ == "__main__":
    main()
