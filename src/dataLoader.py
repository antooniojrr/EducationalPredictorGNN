from datetime import datetime
import os
import re
import numpy as np
import pandas as pd
import torch
import warnings

warnings.simplefilter('ignore', DeprecationWarning)


ASISTENCIA_CSV = "asistencia_sin_ss.csv"
SEGUIMIENTO_CSV = "seguimiento.csv"
ENCUESTAS_DIR = "encuestas/"
PREDECIR_CSV = "predecir.csv"
REL_PATH_DATA = "./data/"



"""Clase para cargar y procesar los datos del curso de manera estructurada en un tensor de Torch."""
class DataLoader:
    CAT_OPTIONS = ['Temp', 'MP', 'Concat']
    
    FEATURE_NAMES = [
        'Asistencia', 'Nota_Semanal',
        'Encuesta_P1', 'Encuesta_P2', 'Encuesta_P3', 'Encuesta_P4', 'Encuesta_P5','Encuesta_P6',
        'Compromiso', 'Estrés'
    ]

    def __init__(self, data_dir=REL_PATH_DATA,num_weeks=12, questions_per_survey=6, start_date=datetime(2025, 2, 27)):
        """
        root_dir: Ruta donde están tus csv
        num_weeks: Duración total del curso en semanas (para dimensionar el tensor)
        """
        self.root = data_dir
        self.num_weeks = num_weeks
        self.questions_per_survey = questions_per_survey
        self.start_date = start_date    # Fecha de la primera clase de la asignatura (De asistencia.csv)
        self.students_ids = None        # Se llenará al cargar

        self.week_data = {}

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"❌ LA CARPETA NO EXISTE: {os.path.relpath(self.root)}\n")
        os.makedirs(os.path.join(self.root, 'processed/tensors'), exist_ok=True)
    
    def _parse_date(self, date_str):
        """Intenta parsear fechas en formatos dd/mm o dd_mm"""
        for fmt in ('%d/%m', '%d_%m'):
            try:
                dt = datetime.strptime(date_str, fmt)
                # Asignamos el año en el que se realizó, aunque no es necesario
                return dt.replace(year=self.start_date.year) 
            except ValueError:
                continue
        return None
     
    def _get_week_index(self, date_obj):
        """Devuelve el índice de la semana (0 a num_weeks) para una fecha dada"""
        if date_obj is None: return -1

        dif = date_obj - self.start_date
        week_idx = dif.days // 7
        return max(0, min(week_idx, self.num_weeks - 1))
    
    def _load_IDs(self):
        df_target = pd.read_csv(os.path.join(self.root, PREDECIR_CSV), decimal=',')
        return df_target['ID'].values
    
    def _load_target(self):
        df_target = pd.read_csv(os.path.join(self.root, PREDECIR_CSV), decimal=',')

        # Target: Nota final normalizada
        y = torch.tensor(df_target['final'].values / 10.0, dtype=torch.float).unsqueeze(1)

        return y
    
    def _load_attendance(self):
        # Asumimos que las columnas de asistencia están ordenadas cronológicamente
        df_att = pd.read_csv(os.path.join(self.root, ASISTENCIA_CSV))
        
        # Comprobar si están ordenados los índices
        df_att = df_att.set_index('ID').reindex(self.students_ids).reset_index()

        att_tensor = np.ones((len(self.students_ids), self.num_weeks)) # 1 = Presente por defecto
        
        # Filtrar solo columnas de asistencia
        att_cols = [c for c in df_att.columns if 'asistencia' in c.lower()]
        
        for week_idx, col in enumerate(att_cols):
            # Solo procesamos hasta el límite de semanas definido en num_weeks
            if week_idx < self.num_weeks:
                # Mapear valores. NO = 0, cualquier otro = 1
                vals = df_att[col].fillna('PRESENT').astype(str)
                is_absent = vals.str.contains('NO', case=False)
                
                att_tensor[:, week_idx] = np.where(is_absent, 0.0, 1.0)

        return att_tensor
    
    def _load_grades(self):
        df_grades = pd.read_csv(os.path.join(self.root, 'seguimiento.csv'), decimal=',')
        
        # --- ALINEACIÓN DE SEGURIDAD ---
        df_grades = df_grades.set_index('ID').reindex(self.students_ids).reset_index()
        
        grades_tensor = np.zeros((len(self.students_ids), self.num_weeks)) # 0 por defecto
        
        for col in df_grades.columns:
            col_lower = col.lower()
            
            # --- FILTRO: SOLO PRÁCTICAS ---
            # Si la columna contiene 'examen', saltamos a la siguiente
            if 'examen' in col_lower:
                continue
                

            # --- PROCESAMIENTO DE FECHA Y VALOR ---
            # Buscar fecha en paréntesis, ej: P1_(23_03)
            match = re.search(r'\((\d{1,2}[_/]\d{1,2})\)', col)
            if match:
                date_str = match.group(1)
                dt = self._parse_date(date_str)
                week = self._get_week_index(dt)
                
                # Limpiar notas (NP -> 0)
                vals = df_grades[col].replace('NP', 0).fillna(0)
                
                # Asegurar que sean float
                vals = vals.astype(str).str.replace(',', '.').astype(float)
                vals = vals / 10.0  # Normalizar a [0, 1]
                
                # Solo guardamos si la semana cae dentro del curso
                if week >= 0 and week < self.num_weeks:
                    # Si en una misma semana hay dos entregas (ej: P2 Indiv y P2 Grupal),
                    # sumamos las notas. 
                    grades_tensor[:, week] += vals.values

        return grades_tensor
    
    def _load_surveys(self):
        # Tensor shape: [N_students, N_weeks, 6_features]
        surveys_tensor = np.zeros((len(self.students_ids), self.num_weeks, self.questions_per_survey + 2)) # +2 para features calculadas
        
        # Buscamos los archivos de las encuestas
        survey_files = [f for f in os.listdir(os.path.join(self.root, ENCUESTAS_DIR))]
        
        for f in survey_files:
            # Extraer índice de semana del nombre
            try:
                week_num = int(f.split('.')[0]) # Coge el número antes del primer punto
                week_idx = week_num             # Empiezan en la semana 1
            except ValueError:
                continue # Si el nombre no cumple formato, saltamos

            if 0 <= week_idx < self.num_weeks:
                df_s = pd.read_csv(os.path.join(self.root,ENCUESTAS_DIR, f), decimal=',')
                
                # --- ALINEACIÓN DE SEGURIDAD ---
                df_s = df_s.set_index('ID').reindex(self.students_ids).reset_index()
                
                # PREGUNTAS 1-5
                # Seleccionamos por posición (iloc) para no depender de los nombres largos
                # Columna 0 es ID, así que cogemos de la 1 a la 5 (indices 1:6) y los que no hayan contestado se ponen en 0
                numeric_cols = df_s.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').fillna(0)
                # Normalizamos a rango [0, 1] dividiendo por 5
                numeric_vals = numeric_cols.values / 5.0
                
                # PREGUNTA SI/NO/A MEDIAS ---
                text_col = df_s.iloc[:, 6].astype(str).fillna('No')
                text_numeric = np.zeros(len(text_col))
                
                # 1. Identificar "Si" (Alerta Roja) -> 1.0
                is_yes = text_col.str.contains('Si', case=False, na=False)
                text_numeric[is_yes] = 1.0
                
                # 2. Identificar "No" (Todo bien) -> 0.0
                is_no = text_col.str.contains('No', case=False, na=True)
                
                # 3. Identificar "A medias" u otros -> 0.5
                is_intermediate = ~is_yes & ~is_no
                text_numeric[is_intermediate] = 0.5
                

                # AÑADIMOS PARÁMETROS CALCULADOS A PARTIR DE LOS OTROS
                # Compromiso con la asignatura
                engagement_score = np.mean(numeric_vals[:, 0:3], axis=1)

                # Nivel de estrés percibido
                stress_score = np.mean(numeric_vals[:, 3:5], axis=1)

                # Concatenamos
                # Reshape text_numeric de (N,) a (N, 1) para poder pegar
                week_features = np.hstack([numeric_vals, text_numeric.reshape(-1, 1), engagement_score.reshape(-1, 1), stress_score.reshape(-1, 1)])
                
                surveys_tensor[:, week_idx, :] = week_features

        return surveys_tensor
    
    def load_data(self, cat_opt=None) -> tuple[str,torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Carga los datos y estructura el tensor X según cat_opt.
        
        Opciones de cat_opt:
        - 'Temp':      Mantiene la dimensión temporal. Shape: [N, Weeks, F]
                           Ideal para: STGNN, LSTM, GRU.
        - 'MP':  Colapsa el tiempo haciendo la media. Shape: [N, F]
                           Ideal para: GCN, GAT (Modelos estáticos).
        - 'Concat': Aplana el tiempo concatenando semanas. Shape: [N, Weeks*F]
                           Ideal para: MLP, SVM, RF (Modelos clásicos o estáticos).
        """
        
        while not cat_opt:
            cat_opt = self._select_cat_opt()

        print("\n\nCARGANDO DATOS CON OPCIÓN:", cat_opt)
        # 1. CARGAR TARGET e IDs (Notas Finales) --------------------------------------------------
        self.students_ids = self._load_IDs()

        path_y = os.path.join(self.root, 'processed/tensors', 'Y.pt')
        if os.path.exists(path_y):
            print("\t>>> Cargando Y desde archivo procesado...")
            self.Y = torch.load(path_y)
        else:
            print("\t>>> Cargando y procesando Y (Notas Finales)...")
            self.Y = self._load_target()
            os.makedirs(os.path.join(self.root, 'processed/tensors'), exist_ok=True)
            torch.save(self.Y, path_y)

        # 2. CARGAR ASISTENCIA --------------------------------------------------------------
        path_att = os.path.join(self.root, 'processed/tensors', 'attendance.pt')
        if os.path.exists(path_att):
            print("\t>>> Cargando Asistencia desde archivo procesado...")
            att_tensor = torch.load(path_att).numpy()
        else:
            print("\t>>> Cargando y procesando Asistencia...")
            att_tensor = self._load_attendance()
            torch.save(torch.tensor(att_tensor, dtype=torch.float), path_att)

        # 3. CARGAR SEGUIMIENTO (Notas parciales) ------------------------------------------
        path_grades = os.path.join(self.root, 'processed/tensors', 'grades.pt')
        if os.path.exists(path_grades):
            print("\t>>> Cargando Notas Parciales desde archivo procesado...")
            grades_tensor = torch.load(path_grades).numpy()
        else:
            print("\t>>> Cargando y procesando Notas Parciales...")
            grades_tensor = self._load_grades()
            torch.save(torch.tensor(grades_tensor, dtype=torch.float), path_grades)

        # 4. CARGAR ENCUESTAS (Iterar por archivos)--------------------------------------------
        path_surveys = os.path.join(self.root, 'processed/tensors', 'surveys.pt')
        if os.path.exists(path_surveys):
            print("\t>>> Cargando Encuestas desde archivo procesado...")
            surveys_tensor = torch.load(path_surveys).numpy()
        else:
            print("\t>>> Cargando y procesando Encuestas...")
            surveys_tensor = self._load_surveys()
            torch.save(torch.tensor(surveys_tensor, dtype=torch.float), path_surveys)
        
        # 5. CONSOLIDAR TENSOR X (Concatenar features)-------------------------------------
        # Base: [N, Weeks, F_total]
        # F_total = 1 (Asistencia) + 1 (Nota Semanal) + N_preguntas + 2
        
        # Expandir dimensiones para concatenar
        att_expanded = torch.tensor(att_tensor, dtype=torch.float).unsqueeze(2)
        grades_expanded = torch.tensor(grades_tensor, dtype=torch.float).unsqueeze(2)
        surv_expanded = torch.tensor(surveys_tensor, dtype=torch.float)

        path_x = os.path.join(self.root, 'processed/tensors', "X_"+cat_opt+".pt")
        if os.path.exists(path_x):
            print(f"\t>>> Cargando X ({cat_opt}) desde archivo procesado...")
            self.X = torch.load(path_x)
            raw_comps = [att_expanded, grades_expanded, surv_expanded]
            print(f"Dimensiones finales de X: {self.X.shape}, Y: {self.Y.shape}")
            print(f"Dimensiones componentes crudas: {[comp.shape for comp in raw_comps]}")
            return cat_opt, self.X, self.Y, raw_comps
        
        else:
            print(f"\t>>> Cargando y procesando datos (Modo: {cat_opt})...")
            
            # Creamos el tensor base con estructura Temporal [N, Weeks, Features]
            X_base = torch.cat([att_expanded, grades_expanded, surv_expanded], dim=2)

            # --- APLICAR TRANSFORMACIÓN SEGÚN cat_opt ---
            if cat_opt == 'Temp':
                # No hacemos nada, mantenemos [N, Weeks, F]
                self.X = X_base
                
            elif cat_opt == 'MP':
                # Hacemos la media a lo largo del eje temporal (dim=1)
                # [N, Weeks, F] -> [N, F]
                self.X = torch.mean(X_base, dim=1)
                
            elif cat_opt == 'Concat':
                # Aplanamos las semanas y las features en un solo vector largo por alumno
                # [N, Weeks, F] -> [N, Weeks * F]
                # .reshape(N, -1) le dice a Torch: mantén N filas y calcula el resto automáticamente
                self.X = X_base.reshape(X_base.shape[0], -1)
            
            torch.save(self.X, os.path.join(self.root, 'processed/tensors', "X_"+cat_opt+".pt"))
            raw_comps = [att_expanded, grades_expanded, surv_expanded]

            print(f"Dimensiones finales de X: {self.X.shape}, Y: {self.Y.shape}")
            print(f"Dimensiones componentes crudas: {[comp.shape for comp in raw_comps]}")
            return cat_opt, self.X, self.Y, raw_comps
    
    def get_X_from_file(self, cat_opt='Temporal'):
        """Carga el tensor X desde archivo procesado."""
        path_x = os.path.join(self.root, 'processed', 'tensors', "X_"+cat_opt+".pt")
        if not os.path.exists(path_x):
            raise FileNotFoundError(f"❌ No se encontró el archivo X en {path_x}. "
                                    "Ejecuta load_data() primero para generarlo.")
        self.X = torch.load(path_x)
        return self.X
    
    def get_Y_from_file(self):
        """Carga el tensor Y desde archivo procesado."""
        path_y = os.path.join(self.root, 'processed', 'tensors', 'Y.pt')
        if not os.path.exists(path_y):
            raise FileNotFoundError(f"❌ No se encontró el archivo Y en {path_y}. "
                                    "Ejecuta load_data() primero para generarlo.")
        self.Y = torch.load(path_y)
        return self.Y
    
    def _select_cat_opt(self):
        """Función para listar las opciones de cat_opt y permitir al usuario seleccionar una."""
        print("Opciones de cat_opt disponibles:")
        for i, opt in enumerate(self.CAT_OPTIONS):
            print(f"{i+1}. {opt}")
        
        choice = int(input("Selecciona el número de la opción que quieres usar: ")) - 1
        if 0 <= choice < len(self.CAT_OPTIONS):
            return self.CAT_OPTIONS[choice]
        else:
            print("Selección inválida.")
            return None
    
if __name__ == "__main__":
    # Prueba rápida de carga
    loader = DataLoader()
    cat_opt, X, Y, raw_comps = loader.load_data()
    # Imprimir todos los tensores cargados
    print("Tensores cargados:")
    print(f"cat_opt: {cat_opt}")
    print(f"X shape: {X.shape}")
    print(X)
    print(f"Y shape: {Y.shape}")
    print(Y)
    