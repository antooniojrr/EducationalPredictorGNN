
import torch
from model import AdaptiveModel
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, mean_absolute_percentage_error
import os

# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

PATH_MODELS = './data/models/'
os.makedirs(PATH_MODELS, exist_ok=True)

class EntrenadorGNN:
    """Entrenar y evaluar modelos GNN de forma robusta"""
    DEFAULT_CONFIG = {
        'epochs': 500,
        'lr': 0.01,
        'paciencia': 50,
        'max_restarts': 3,
        
        'model_name': 'STGNN_Model',
        'model_type': 'STGNN',
        'input_dim': None,  # Se establecerá dinámicamente
        'output_dim': 1,
        'hidden_dim': 32,  
        'dropout': 0.2,
        'num_layers': 2,
        'type_stgnn': 'GAT',
        'flexible': False
    }

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {self.device}\n")
    
    def shake_weights(self, model, std=0.01):
        """Añade ruido gaussiano a los pesos para escapar de mínimos locales"""
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * std
                param.add_(noise)
    
    def entrenar(self, data, train_idx = None, test_idx = None, config=None, verbose=True):
        if config and config.get('flexible', False):
            return self.entrenar_flexible(data, train_idx, test_idx, config, verbose=verbose)
        else:
            return self.entrenar_rigido(data, train_idx, test_idx, config, verbose=verbose)
        
    def entrenar_rigido(self, data, train_idx = None, test_idx = None, 
                          config=None, verbose=True):
        """
        Args:
            data: Objeto Data de PyG.
            train_idx, test_idx: Tensores con los índices FIJOS.
            config: Diccionario con hiperparámetros del modelo y de entrenamiento (Opcional).
        """
        
        if train_idx is None or test_idx is None:
            # Crear índices de entrenamiento y prueba si no se proporcionan
            num_nodes = data.num_nodes
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            split = int(0.8 * num_nodes)
            train_idx = torch.tensor(indices[:split], dtype=torch.long)
            test_idx = torch.tensor(indices[split:], dtype=torch.long)

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

        # Configuración por defecto si no se pasa nada
        cfg = self.DEFAULT_CONFIG.copy()
        cfg['input_dim'] = input_dim
        if config: cfg.update(config)
            
        modelo = AdaptiveModel(
            model_type=cfg['model_type'],
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'], 
            output_dim=cfg['output_dim'],
            num_layers=cfg['num_layers'], 
            dropout=cfg['dropout'],
            type_stgnn=cfg['type_stgnn']
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
        
        if verbose: print(f"🚀 Iniciando training: {cfg['model_name']} para {cfg['epochs']} epochs...")
        
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
                y_pred_test = pred[test_idx].cpu().numpy().flatten() * 10
                y_true_test = data.y[test_idx].cpu().numpy().flatten() * 10
                
                current_r2 = r2_score(y_true_test, y_pred_test)

                #scheduler.step(current_r2)
            # --- LOGGING & EARLY STOPPING ---
            if epoch % 10 == 0:
                test_mae = torch.mean(torch.abs(pred[test_idx] - data.y[test_idx])).item()
                train_mae = torch.mean(torch.abs(pred[train_idx] - data.y[train_idx])).item()
                if verbose: print(f"Epoch {epoch:3d} | R2: {current_r2:.4f} | Train Loss: {loss.item():.4f} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")

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
                        if verbose: print(f"🛑 Paciencia agotada en epoch {epoch}. Intentando revivir...")
                        
                        # 1. Cargar el mejor estado hasta ahora (para no agitar basura)
                        if mejor_modelo_state:
                            modelo.load_state_dict(mejor_modelo_state)
                        
                        # 2. Hacer Shake (añadir ruido)
                        self.shake_weights(modelo, std=0.08 * (0.8 ** restarts_done)) # Ruido decreciente
                        if verbose: print(f"🫨 SHAKE! Ruido inyectado a los pesos")
                        # 3. Reiniciar optimizador con LR más bajo
                        current_lr = optimizer.param_groups[0]['lr']
                        new_lr = current_lr * 0.5
                        optimizer = torch.optim.Adam(modelo.parameters(), lr=new_lr, weight_decay=1e-4)
                        
                        if verbose: print(f"🔄 RESTART #{restarts_done+1}: LR bajado a {new_lr:.5f}, Paciencia reseteada.")

                        contador_paciencia = 0
                        restarts_done += 1
                    else:
                        if verbose: print(f"❌ Máximos restarts alcanzados. Terminando entrenamiento en epoch {epoch}.")
                        break
        
        # --- RESTAURAR MEJOR MODELO ---
        if mejor_modelo_state:
            modelo.load_state_dict(mejor_modelo_state)
        if verbose: print(f"✅ Entrenamiento finalizado. Mejor R2 en Test: {mejor_metric_test:.4f}")

        # --- REPORTE FINAL ---
        if verbose: print("\n📊 Evaluación Final del Mejor Modelo.")
        modelo.eval()
        with torch.no_grad():
            pred = modelo(data)
            
            train_pred = pred[train_idx].cpu().numpy().flatten()
            train_true = data.y[train_idx].cpu().numpy().flatten()
            test_pred = pred[test_idx].cpu().numpy().flatten()
            test_true = data.y[test_idx].cpu().numpy().flatten()
            
            return self.calculate_metrics(test_true, test_pred, threshold=0.7), test_true, test_pred, modelo, cfg
    
    def entrenar_flexible(self, data, train_idx = None, test_idx = None, config=None, verbose=True):
        """
        Args:
            data: Objeto Data de PyG.
            train_idx, test_idx: Tensores con los índices FIJOS.
            config: Diccionario con hiperparámetros del modelo y de entrenamiento (Opcional).
        """
        if train_idx is None or test_idx is None:
            # Crear índices de entrenamiento y prueba si no se proporcionan
            num_nodes = data.num_nodes
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            split = int(0.8 * num_nodes)
            train_idx = torch.tensor(indices[:split], dtype=torch.long)
            test_idx = torch.tensor(indices[split:], dtype=torch.long)

        # Mover datos al dispositivo
        data = data.to(self.device)
        train_idx = train_idx.to(self.device)
        test_idx = test_idx.to(self.device)

        # Instanciar Modelo Dinámicamente
        if data.x.dim() != 3:
            raise ValueError("Los datos de entrada deben tener dimensión 3 para entrenamiento flexible (N, Semanas, Features).")
        
        input_dim = data.x.size(2)

        # Configuración por defecto si no se pasa nada
        cfg = self.DEFAULT_CONFIG.copy()
        cfg['input_dim'] = input_dim
        if config: cfg.update(config)
            
        modelo = AdaptiveModel(
            model_type=cfg['model_type'],
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'], 
            output_dim=cfg['output_dim'],
            num_layers=cfg['num_layers'], 
            dropout=cfg['dropout'],
            type_stgnn=cfg['type_stgnn']
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
        
        if verbose:print(f"🚀 Iniciando training con VENTANA ALEATORIA: {cfg['model_name']} para {cfg['epochs']} epochs...")

        for epoch in range(cfg['epochs']):
            modelo.train()
            optimizer.zero_grad()
            
            # Crear batch con ventana aleatoria
            data_batch = self.randomly_slice_data(data)

            out = modelo(data_batch)
            
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
                y_pred_test = pred[test_idx].cpu().numpy().flatten() * 10
                y_true_test = data.y[test_idx].cpu().numpy().flatten() * 10
                
                current_r2 = r2_score(y_true_test, y_pred_test)

                #scheduler.step(current_r2)
            # --- LOGGING & EARLY STOPPING ---
            if epoch % 10 == 0:
                test_mae = torch.mean(torch.abs(pred[test_idx] - data.y[test_idx])).item()
                train_mae = torch.mean(torch.abs(pred[train_idx] - data.y[train_idx])).item()
                if verbose:print(f"Epoch {epoch:3d} | R2: {current_r2:.4f} | Train Loss: {loss.item():.4f} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")

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
                        if verbose: print(f"🛑 Paciencia agotada en epoch {epoch}. Intentando revivir...")
                        
                        # 1. Cargar el mejor estado hasta ahora (para no agitar basura)
                        if mejor_modelo_state:
                            modelo.load_state_dict(mejor_modelo_state)
                        
                        # 2. Hacer Shake (añadir ruido)
                        self.shake_weights(modelo, std=0.08 * (0.8 ** restarts_done)) # Ruido decreciente
                        
                        # 3. Reiniciar optimizador con LR más bajo
                        current_lr = optimizer.param_groups[0]['lr']
                        new_lr = current_lr * 0.5
                        optimizer = torch.optim.Adam(modelo.parameters(), lr=new_lr, weight_decay=1e-4)
                        
                        if verbose: print(f"🔄 RESTART #{restarts_done+1}: LR bajado a {new_lr:.5f}, Paciencia reseteada.")

                        contador_paciencia = 0
                        restarts_done += 1
                    else:
                        if verbose: print(f"❌ Máximos restarts alcanzados. Terminando entrenamiento en epoch {epoch}.")
                        break

        # --- RESTAURAR MEJOR MODELO ---
        if mejor_modelo_state:
            modelo.load_state_dict(mejor_modelo_state)
        if verbose: print(f"✅ Entrenamiento finalizado. Mejor R2 en Test: {mejor_metric_test:.4f}")

        # --- REPORTE FINAL ---
        if verbose: print("\n📊 Evaluación Final del Mejor Modelo.")
        modelo.eval()
        with torch.no_grad():
            pred = modelo(data)
            
            train_pred = pred[train_idx].cpu().numpy().flatten()
            train_true = data.y[train_idx].cpu().numpy().flatten()
            test_pred = pred[test_idx].cpu().numpy().flatten()
            test_true = data.y[test_idx].cpu().numpy().flatten()
            
            return self.calculate_metrics(test_true, test_pred, threshold=0.7), test_true, test_pred, modelo, cfg
        
    def randomly_slice_data(self, data, min_weeks=4):    
        """
        Corta aleatoriamente las features y el grafo hasta una semana aleatoria entre min_weeks y total_weeks.
        """

        total_weeks = data.x.shape[1] 
        dyn_edges_all = getattr(data, "dynamic_edge_indices", None)
        # En cada época, el modelo ve una longitud distinta
        # np.random.randint(min_weeks, total_weeks + 1) devuelve un entero entre min_weeks y total_weeks
        cut_t = np.random.randint(min_weeks, total_weeks + 1)
        
        # 1. Recortamos las Features [N, cut_t, F]
        x_cut = data.x[:, :cut_t, :]
        
        # 2. Gestionamos el Grafo para este recorte
        if dyn_edges_all is not None:
            # Si tenemos grafos dinámicos, pasamos la lista recortada
            # El modelo STGNN sabrá usar el grafo 't' para la semana 't'
            current_dyn_edges = dyn_edges_all[:cut_t]
            current_static_edge = dyn_edges_all[cut_t-1] # El último disponible
        else:
            # Si es estático, usamos el grafo estático (aproximación)
            # OJO: Esto introduce un poco de ruido (leakage), pero en ventana aleatoria es aceptable
            current_dyn_edges = None
            current_static_edge = data.edge_index

        # Creamos un objeto Data temporal para este paso
        # IMPORTANTE: data.y NO se recorta. Queremos predecir la nota FINAL con datos PARCIALES.
        data_batch = Data(x=x_cut, edge_index=current_static_edge, y=data.y)
        if current_dyn_edges:
            data_batch.dynamic_edge_indices = current_dyn_edges
        
        data_batch = data_batch.to(self.device)
    
        return data_batch
    
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
        mape = mean_absolute_percentage_error(y_true_10, y_pred_10)
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
            "MAPE": mape,
            "R2": r2,
            "Accuracy": acc,
            "F1_Score": f1,
            "True_Negatives": tn, # Predijo suspenso, fue suspenso (Bien)
            "False_Negatives": fn, # Predijo suspenso, fue aprobado (Falsa Alarma)
            "False_Positives": fp, # Predijo aprobado, fue suspenso (PELIGRO)
            "True_Positives": tp   # Predijo aprobado, fue aprobado (Bien)
        }
        
        return metrics
    
    def save_model(self, model, cfg, metrics = None, dir=None):
        path = PATH_MODELS
        if dir:
            path = path + dir + "/"
            os.makedirs(path, exist_ok=True)

        model_name = cfg['model_name']
        if cfg.get('flexible', False):
            model_name += "_flexible"

        model_path = os.path.join(path, f"{model_name}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Modelo guardado en: {model_path}")

        # Guardo también la configuración usada
        config_path = os.path.join(path, f"{model_name}_config.txt")
        with open(config_path, 'w') as f:
            f.write(f"Configuracion del Modelo {cfg['model_name']}:\n")
            for key, value in cfg.items():
                if key != 'stgnn_type' or cfg['model_type'] == 'STGNN':
                    f.write(f"{key}: {value}\n")
        print(f"💾 Configuración guardada en: {config_path}")

        # Guardo las métricas en un archivo separado
        if metrics:
            metrics_path = os.path.join(path, f"{model_name}_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Metricas del Modelo {cfg['model_name']}:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            print(f"💾 Métricas guardadas en: {metrics_path}")

    def load_model(self, model_name, flexible=False, dir=None):
        # Leemos la configuración del archivo correspondiente
        path = PATH_MODELS
        if dir:
            path = path + dir + "/"
        if flexible:
            model_name += "_flexible"
        config_path = os.path.join(path, f"{model_name}_config.txt")
        cfg = {}
        with open(config_path, 'r') as f:
            lines = f.readlines()[1:]  # Saltamos la primera línea de título
            for line in lines:
                key, value = line.strip().split(': ')
                # Convertimos a int o float si es posible
                if value.replace('.', '', 1).isdigit():
                    if '.' in value:
                        cfg[key] = float(value)
                    else:
                        cfg[key] = int(value)
                else:
                    cfg[key] = value
        
        # Instanciamos el modelo con la configuración leída
        modelo = AdaptiveModel(
            model_type=cfg['model_type'],
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'], 
            output_dim=cfg['output_dim'],
            num_layers=cfg['num_layers'], 
            dropout=cfg['dropout'],
            type_stgnn=cfg.get('type_stgnn', 'GAT')  # Valor por defecto si no existe
        ).to(self.device)

        # Cargamos los pesos guardados
        model_path = os.path.join(path, f"{model_name}_model.pth")
        modelo.load_state_dict(torch.load(model_path, map_location=self.device))

        print(f"✅ Modelo {model_name} cargado desde {model_path}")
        return modelo, cfg


    