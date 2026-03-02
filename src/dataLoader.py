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


class DataLoader:
    """
    Clase encargada de cargar, alinear y estructurar los datos académicos del curso
    (asistencia, seguimiento y encuestas) en tensores de PyTorch preparados para
    su uso en modelos de aprendizaje automático.

    Permite generar distintas representaciones del tensor de entrada según la
    estrategia de agregación temporal seleccionada.
    """

    CAT_OPTIONS = ['Temp', 'MP', 'Concat']
    
    FEATURE_NAMES = [
        'Asistencia', 'Nota_Semanal',
        'Encuesta_P1', 'Encuesta_P2', 'Encuesta_P3', 'Encuesta_P4', 'Encuesta_P5','Encuesta_P6',
        'Compromiso', 'Estrés'
    ]

    def __init__(self, data_dir=REL_PATH_DATA, num_weeks=12, questions_per_survey=6, start_date=datetime(2025, 2, 27)):
        """
        Inicializa el cargador de datos.

        Args:
            data_dir (str): Directorio raíz donde se encuentran los archivos CSV.
            num_weeks (int): Número total de semanas consideradas en el curso.
            questions_per_survey (int): Número de preguntas numéricas por encuesta.
            start_date (datetime): Fecha de inicio del curso, utilizada para calcular
                                   el índice temporal de cada actividad.
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
        """
        Convierte una cadena de fecha en formato 'dd/mm' o 'dd_mm' en un objeto datetime.

        Args:
            date_str (str): Fecha en formato abreviado sin año.

        Returns:
            datetime | None: Fecha con el año del curso asignado si el parseo es válido;
                             None en caso contrario.
        """
        for fmt in ('%d/%m', '%d_%m'):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(year=self.start_date.year) 
            except ValueError:
                continue
        return None
     
    def _get_week_index(self, date_obj):
        """
        Calcula el índice de semana correspondiente a una fecha dada
        respecto a la fecha de inicio del curso.

        Args:
            date_obj (datetime): Fecha a convertir en índice semanal.

        Returns:
            int: Índice de semana acotado al rango [0, num_weeks-1].
                 Devuelve -1 si la fecha es None.
        """
        if date_obj is None: return -1

        dif = date_obj - self.start_date
        week_idx = dif.days // 7
        return max(0, min(week_idx, self.num_weeks - 1))
    
    def _select_cat_opt(self):
        """
        Solicita al usuario la selección del modo de agregación temporal.

        Returns:
            str | None: Opción válida dentro de CAT_OPTIONS o None si la selección es inválida.
        """

        print("A continuación se presentan las opciones de agregación temporal (cat_opt):")
        for i, opt in enumerate(self.CAT_OPTIONS):
            print(f"{i+1}. {opt}")
        
        choice = int(input("Selecciona el número de la opción que quieres usar: ")) - 1
        if 0 <= choice < len(self.CAT_OPTIONS):
            return self.CAT_OPTIONS[choice]
        else:
            print("Selección no válida. Por favor, inténtelo de nuevo.")
            return None

    def _load_IDs(self):
        """
        Carga los identificadores de los estudiantes desde el archivo de predicción.

        Returns:
            np.ndarray: Vector con los IDs de los estudiantes.
        """
        df_target = pd.read_csv(os.path.join(self.root, PREDECIR_CSV), decimal=',')
        return df_target['ID'].values
    
    def _load_target(self):
        """
        Carga la variable objetivo (nota final) y la normaliza al rango [0, 1].

        Returns:
            torch.Tensor: Tensor columna de dimensión [N, 1] con las notas finales normalizadas.
        """
        df_target = pd.read_csv(os.path.join(self.root, PREDECIR_CSV), decimal=',')

        y = torch.tensor(df_target['final'].values / 10.0, dtype=torch.float).unsqueeze(1)

        return y
    
    def _load_attendance(self):
        """
        Carga los datos de asistencia semanal y los transforma en una matriz numérica.

        La asistencia se codifica como:
            1.0 → Presente
            0.0 → Ausente

        Returns:
            np.ndarray: Matriz de dimensión [N_estudiantes, num_weeks].
        """

        # Asumimos que las columnas de asistencia están ordenadas cronológicamente
        df_att = pd.read_csv(os.path.join(self.root, ASISTENCIA_CSV))
        
        # Comprobar si están ordenados los índices
        df_att = df_att.set_index('ID').reindex(self.students_ids).reset_index()

        att_tensor = np.ones((len(self.students_ids), self.num_weeks)) # 1 = Presente por defecto
        
        # Filtrar solo columnas de asistencia
        att_cols = [c for c in df_att.columns if 'asistencia' in c.lower()]
        
        for week_idx, col in enumerate(att_cols):
            if week_idx < self.num_weeks:
                #NO = 0, cualquier otro = 1
                vals = df_att[col].fillna('PRESENT').astype(str)
                is_absent = vals.str.contains('NO', case=False)
                
                att_tensor[:, week_idx] = np.where(is_absent, 0.0, 1.0)

        return att_tensor
    
    def _load_grades(self):
        """
        Carga las notas parciales semanales y las asigna a la semana correspondiente
        según la fecha contenida en el nombre de la columna.

        Se excluyen columnas asociadas a exámenes.
        Las notas se normalizan al rango [0, 1].

        Returns:
            np.ndarray: Matriz [N_estudiantes, num_weeks] con la suma de notas por semana.
        """
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
        """
        Carga las encuestas semanales, normaliza sus valores y calcula
        métricas agregadas de compromiso y estrés.

        Returns:
            np.ndarray: Tensor [N_estudiantes, num_weeks, questions_per_survey + 2]
                        con preguntas normalizadas y métricas derivadas.
        """
        surveys_tensor = np.zeros((len(self.students_ids), self.num_weeks, self.questions_per_survey + 2)) # +2 para features calculadas
        
        survey_files = [f for f in os.listdir(os.path.join(self.root, ENCUESTAS_DIR))]
        
        for f in survey_files:
            # Extraer índice de semana del nombre
            try:
                week_num = int(f.split('.')[0]) 
                week_idx = week_num            
            except ValueError:
                continue

            if 0 <= week_idx < self.num_weeks:
                df_s = pd.read_csv(os.path.join(self.root,ENCUESTAS_DIR, f), decimal=',')
                
                df_s = df_s.set_index('ID').reindex(self.students_ids).reset_index()
                
                # PREGUNTAS 1-5
                numeric_cols = df_s.iloc[:, 1:6].apply(pd.to_numeric, errors='coerce').fillna(0)
                numeric_vals = numeric_cols.values / 5.0
                
                # PREGUNTA SI/NO/A MEDIAS ---
                text_col = df_s.iloc[:, 6].astype(str).fillna('No')
                text_numeric = np.zeros(len(text_col))
                
               # "Si" (Alerta Roja) -> 1.0
                is_yes = text_col.str.contains('Si', case=False, na=False)
                text_numeric[is_yes] = 1.0
                
                # "No" (Todo bien) -> 0.0
                is_no = text_col.str.contains('No', case=False, na=True)
                
                # "A medias" u otros -> 0.5
                is_intermediate = ~is_yes & ~is_no
                text_numeric[is_intermediate] = 0.5
                

                # PARÁMETROS CALCULADOS
                engagement_score = np.mean(numeric_vals[:, 0:3], axis=1)
                stress_score = np.mean(numeric_vals[:, 3:5], axis=1)

                week_features = np.hstack([numeric_vals, text_numeric.reshape(-1, 1), engagement_score.reshape(-1, 1), stress_score.reshape(-1, 1)])
                
                surveys_tensor[:, week_idx, :] = week_features

        return surveys_tensor
    
    def load_data(self, cat_opt=None) -> tuple[str,torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Ejecuta el pipeline completo de carga, alineación, procesamiento y estructuración
        de los datos.

        Dependiendo de la opción `cat_opt`, genera distintas representaciones del
        tensor de entrada:

            - 'Temp':   Representación temporal [N, Weeks, F].
            - 'MP':     Media temporal [N, F].
            - 'Concat': Concatenación temporal [N, Weeks*F].

        Args:
            cat_opt (str | None): Estrategia de agregación temporal.

        Returns:
            tuple:
                - cat_opt (str): Opción finalmente utilizada.
                - X (torch.Tensor): Tensor de características.
                - Y (torch.Tensor): Tensor objetivo.
                - raw_comps (list[torch.Tensor]): Componentes individuales antes de concatenar.
        """
        
        while not cat_opt:
            cat_opt = self._select_cat_opt()

        print(f"Iniciando la carga de datos con la opción: {cat_opt}")
        
        self.students_ids = self._load_IDs()

        path_y = os.path.join(self.root, 'processed/tensors', 'Y.pt')
        if os.path.exists(path_y):
            print("   Cargando tensor objetivo Y desde archivo procesado...")
            self.Y = torch.load(path_y)
        else:
            print("   Cargando y procesando Y (notas finales)...")
            self.Y = self._load_target()
            os.makedirs(os.path.join(self.root, 'processed/tensors'), exist_ok=True)
            torch.save(self.Y, path_y)

        # --------------------------------------------------------------
        path_att = os.path.join(self.root, 'processed/tensors', 'attendance.pt')
        if os.path.exists(path_att):
            print("   Cargando datos de asistencia desde archivo procesado...")
            att_tensor = torch.load(path_att).numpy()
            # Recortamos en función del número de semanas actual
            if att_tensor.shape[1] > self.num_weeks:
                att_tensor = att_tensor[:, :self.num_weeks]
        else:
            print("   Cargando y procesando datos de asistencia...")
            att_tensor = self._load_attendance()
            if self.num_weeks == 12:
                torch.save(torch.tensor(att_tensor, dtype=torch.float), path_att)

        # --------------------------------------------------------------
        path_grades = os.path.join(self.root, 'processed/tensors', 'grades.pt')
        if os.path.exists(path_grades):
            print("   Cargando notas parciales desde archivo procesado...")
            grades_tensor = torch.load(path_grades).numpy()
            # Recortamos en función del número de semanas actual
            if grades_tensor.shape[1] > self.num_weeks:
                grades_tensor = grades_tensor[:, :self.num_weeks]
        else:
            print("   Cargando y procesando notas parciales...")
            grades_tensor = self._load_grades()
            if self.num_weeks == 12:
                torch.save(torch.tensor(grades_tensor, dtype=torch.float), path_grades)

        # --------------------------------------------------------------
        path_surveys = os.path.join(self.root, 'processed/tensors', 'surveys.pt')
        if os.path.exists(path_surveys):
            print("   Cargando datos de encuestas desde archivo procesado...")
            surveys_tensor = torch.load(path_surveys).numpy()
            # Recortamos en función del número de semanas actual
            if surveys_tensor.shape[1] > self.num_weeks:
                surveys_tensor = surveys_tensor[:, :self.num_weeks, :]
        else:
            print("   Cargando y procesando datos de encuestas...")
            surveys_tensor = self._load_surveys()
            if self.num_weeks == 12:
                torch.save(torch.tensor(surveys_tensor, dtype=torch.float), path_surveys)
        
        # --------------------------------------------------------------
        att_expanded = torch.tensor(att_tensor, dtype=torch.float).unsqueeze(2)
        grades_expanded = torch.tensor(grades_tensor, dtype=torch.float).unsqueeze(2)
        surv_expanded = torch.tensor(surveys_tensor, dtype=torch.float)

        path_x = os.path.join(self.root, 'processed/tensors', "X_"+cat_opt+".pt")
        if os.path.exists(path_x):
            print(f"   Cargando tensor de características X ({cat_opt}) desde archivo procesado...")
            self.X = torch.load(path_x)
            # Recortamos en función del número de semanas actual
            if self.X.shape[1] > self.num_weeks and cat_opt == 'Temp':
                self.X = self.X[:, :self.num_weeks, :]
            raw_comps = [att_expanded, grades_expanded, surv_expanded]
            print(f"Dimensiones finales de X: {self.X.shape}; de Y: {self.Y.shape}")
            print(f"Dimensiones de las componentes originales: {[comp.shape for comp in raw_comps]}")
            return cat_opt, self.X, self.Y, raw_comps
        
        else:
            print(f"   Cargando y procesando datos (modo: {cat_opt})...")
            
            X_base = torch.cat([att_expanded, grades_expanded, surv_expanded], dim=2)

            # --- APLICAR TRANSFORMACIÓN SEGÚN cat_opt ---
            if cat_opt == 'Temp':
                self.X = X_base
                
            elif cat_opt == 'MP':
                # [N, Weeks, F] -> [N, F]
                self.X = torch.mean(X_base, dim=1)
                
            elif cat_opt == 'Concat':
                # [N, Weeks, F] -> [N, Weeks * F]
                self.X = X_base.reshape(X_base.shape[0], -1)
            
            if self.num_weeks == 12:
                torch.save(self.X, os.path.join(self.root, 'processed/tensors', "X_"+cat_opt+".pt"))
            raw_comps = [att_expanded, grades_expanded, surv_expanded]

            print(f"Dimensiones finales de X: {self.X.shape}; de Y: {self.Y.shape}")
            print(f"Dimensiones de las componentes originales: {[comp.shape for comp in raw_comps]}")
            return cat_opt, self.X, self.Y, raw_comps

    def get_X_from_file(self, cat_opt='Temporal'):
        """
        Carga desde disco un tensor X previamente procesado.

        Args:
            cat_opt (str): Modo de agregación temporal utilizado al generar el tensor.

        Returns:
            torch.Tensor: Tensor X almacenado.
        """
        path_x = os.path.join(self.root, 'processed', 'tensors', "X_"+cat_opt+".pt")
        if not os.path.exists(path_x):
            raise FileNotFoundError(f"❌ No se encontró el archivo X en {path_x}. "
                                    "Ejecuta load_data() primero para generarlo.")
        self.X = torch.load(path_x)
        return self.X
    
    def get_Y_from_file(self):
        """
        Carga desde disco el tensor objetivo Y previamente procesado.

        Returns:
            torch.Tensor: Tensor Y almacenado.
        """
        path_y = os.path.join(self.root, 'processed', 'tensors', 'Y.pt')
        if not os.path.exists(path_y):
            raise FileNotFoundError(f"❌ No se encontró el archivo Y en {path_y}. "
                                    "Ejecuta load_data() primero para generarlo.")
        self.Y = torch.load(path_y)
        return self.Y
    
# ___________________________________________________________________________________________________________________

if __name__ == "__main__":
    loader = DataLoader()
    cat_opt, X, Y, raw_comps = loader.load_data()
    print("Se han cargado los tensores:")
    print(f"cat_opt utilizado: {cat_opt}")
    print(f"Forma de X: {X.shape}")
    print(X)
    print(f"Forma de Y: {Y.shape}")
    print(Y)
    