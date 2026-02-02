# 🎓 Predicción de Rendimiento Académico mediante Redes Neuronales de Grafos (GNN)

Este repositorio contiene el código fuente desarrollado para el Trabajo de Fin de Grado (TFG) centrado en la predicción temprana del rendimiento estudiantil. El sistema utiliza técnicas avanzadas de **Deep Learning en Grafos (Graph Deep Learning)** para modelar no solo el historial individual del alumno, sino también la influencia de su entorno social y académico.

## 🚀 Características Principales

El proyecto implementa un pipeline completo de *Machine Learning* educativo, desde el procesamiento de datos crudos hasta la visualización de predicciones.

### 1. Ingeniería de Datos y Grafos (`dataLoader.py`, `graphCreator.py`)
* **Procesamiento Multimodal:** Integra datos de asistencia, calificaciones (seguimiento continuo) y encuestas.
* **Estandarización Robusta:** Normalización para garantizar la convergencia de redes neuronales.
* **Construcción Dinámica de Grafos:** Generación de grafos de estudiantes ($k$-NN) basados en diferentes perfiles de similitud:
    * `a`: Solo Asistencia.
    * `g`: Solo Notas.
    * `a&g`: Híbrido (Asistencia + Notas).
    * `f3w`: Alerta Temprana (Primeras 3 semanas).
* **Dualidad Estática/Temporal:** Soporte para grafos estáticos (snapshot único) y grafos dinámicos (evolución semana a semana).
* **NOTA IMPORTANTE:** No se utilizan las notas de los parciales teóricos de la asignatura ya que con estos son con los que se calcula la nota final. Solo se usan las notas de seguimiento continuo.

### 2. Arquitectura de Modelos (`model.py`)
Implementación modular de modelos de Estado del Arte (SOTA) comparables bajo el mismo framework:
* **Baselines:**
    * `LSTM`: Red recurrente pura (ignora la estructura social).
* **GNNs Estáticas (Spatial):**
    * `GCN` (Graph Convolutional Network).
    * `GAT` (Graph Attention Network) con Multi-Head Attention.
    * `GraphSAGE` (Inductive Representation Learning).
* **GNN Espacio-Temporal (Spatio-Temporal):**
    * `STGNN`: Arquitectura híbrida personalizada que combina convoluciones gráficas frame a frame con una LSTM para capturar la evolución temporal de los embeddings sociales.

### 3. Entrenamiento Robusto y Validación (, `modelTrainer.py`, `modelTester.py`)
* **Validación Cruzada (K-Fold Cross-Validation):** Evaluación rigurosa (k=5) para garantizar la robustez de los resultados.
* **Estrategias Anti-Overfitting:**
    * *Early Stopping*.
    * **Shake & Restart:** Mecanismo avanzado que inyecta ruido a los pesos y reinicia el optimizador si el modelo cae en mínimos locales (colapso a la media).
* **Métricas Completas:** Evaluación simultánea de Regresión ($R^2$, MAE, RMSE) y Clasificación (Accuracy, F1-Score para detección de riesgo).
* **Entrenamiento de Modelos Flexibles:** Puede entrenar modelos para que sean útiles en entradas de datos temporales incompletas (menos semanas).

### 4. Visualización y Análisis (`graphTester.py`, `predictionsVisualizer.py`)
* **Análisis de Homofilia:** Métricas para cuantificar si "los iguales se juntan con iguales" (Assortativity, Dirichlet Energy).
* **Visualización de Grafos:** Generación de GIFs para observar la evolución de las conexiones entre alumnos.
* **Gráficas de Rendimiento:** Scatter plots (Predicción vs Realidad) y Line plots ordenados para diagnosticar el comportamiento del modelo.

### 5. Estudio de Influencia de Features y Weeks (Explainability) (`modelExplainability.py`)
* **Ablation Studies:** Evaluación del impacto de las diferentes features estudiadas y semanas registradas
---

## ⚙️ Configuraciones Probadas

El sistema permite la combinación flexible de diferentes estrategias de entrada y modelado:

| Estrategia de Datos (`cat_opt`) | Descripción | Modelos Compatibles |
| :--- | :--- | :--- |
| **MP (Mean Pooling)** | Promedio de todas las semanas. Visión estática del curso. | GCN, GAT, SAGE |
| **Concat** | Concatenación de todas las semanas en un vector largo. | MLP (implícito en GNNs) |
| **Temp (Temporal)** | Secuencia temporal `[N, Weeks, Features]`. | LSTM, STGNN |

**Perfiles de Similitud para el Grafo:**
* **`a&g` (Asistencia + Notas):** El grafo conecta alumnos con hábitos de asistencia y rendimiento similares. (Configuración por defecto recomendada).

---

## 📊 Resultados Experimentales

A continuación se presentan los resultados obtenidos tras la validación cruzada (5-Folds) en el conjunto de datos final.

*(Copia y pega aquí la tabla que imprime tu script `main.py` al finalizar)*

| Modelo | MAE (Error) ↓ | R² (Explicabilidad) ↑ | Accuracy (Clasif.) ↑ | F1-Score (Riesgo) ↑ |
| :--- | :---: | :---: | :---: | :---: |
**LSTM** | 0.680404 +- 0.081843 | 0.627564 | 0.930476 | 0.929526
**GCN** | 0.953232 +- 0.260395 | 0.276488 | 0.796190 | 0.783408
**GAT** | 0.936292 +- 0.165051 | 0.298808 | 0.822857 | 0.797644
**SAGE** | 0.691374 +- 0.091066 | 0.607332 | 0.890476 | 0.866539
**STGNN** | 0.751133 +- 0.064971 | 0.506790 | 0.877143 | 0.851230

> **Interpretación:**
> * **MAE:** Error medio absoluto en puntos (sobre 10).
> * **R²:** Proporción de la varianza de las notas explicada por el modelo.
> * **F1-Score:** Métrica crítica para evaluar la capacidad de detectar alumnos suspensos sin falsas alarmas.

## ⚙️ Configuración del Entrenamiento y Hiperparámetros Ganadores (Por ahora)

A continuación se detallan los hiperparámetros y configuraciones utilizadas para obtener los resultados experimentales. Estos valores se encuentran definidos en `src/model.py` y `src/main.py`.

### 🎛️ Hiperparámetros Generales
Configuración por defecto del `EntrenadorGNN`:

| Parámetro | Valor | Descripción |
| :--- | :---: | :--- |
| **Épocas (Epochs)** | `500` | Máximo número de iteraciones de entrenamiento. |
| **Learning Rate (LR)** | `0.01` | Tasa de aprendizaje inicial. |
| **Hidden Dimension** | `32` | Tamaño de los vectores de características en capas ocultas. |
| **Num Layers** | `2` | Número de capas de convolución (GNN) o recurrencia. |
| **Dropout** | `0.2` | Probabilidad de desactivación de neuronas (Regularización). |
| **Weight Decay** | `5e-4` | Penalización L2 en el optimizador Adam. |
| **Paciencia (Early Stop)**| `50` | Épocas sin mejora antes de activar *Shake* o detener. |
| **Max Restarts** | `3` | Número máximo de reinicios permitidos. |

### 🧠 Estrategias de Optimización
El sistema implementa mecanismos avanzados para evitar el colapso a la media y los mínimos locales:

* **Optimizador:** `Adam`.
* **Función de Pérdida:** `MSELoss` (Error Cuadrático Medio) sobre salidas normalizadas [0, 1].
* **Scheduler (LR):** `ReduceLROnPlateau` (DESACTIVADO actualmente).
    * Modo: `max` (Maximizar $R^2$).
    * Factor: `0.5` (Reduce LR a la mitad).
    * Paciencia: `10` épocas.
* **Mecanismo "Shake & Restart":**
    * **Vidas Extra:** `3` reinicios permitidos.
    * **Inyección de Ruido:** Ruido gaussiano inicial ($\sigma=0.08$) con decaimiento exponencial ($0.8^n$) en cada reinicio.
    * **Reinicio de LR:** Se reduce el LR actual al 50% tras cada *Shake*.

### 🏗️ Arquitectura de Modelos Específicos
Detalles de configuración interna para cada variante:

* **GAT (Graph Attention Network):**
    * **Heads:** `2` cabezales de atención.
    * Activación: `ELU`.
* **GraphSAGE:**
    * **Agregador:** `LSTM` (requiere ordenación de vecinos).
    * Activación: `ReLU`.
* **STGNN (Espacio-Temporal):**
    * **Backbone Espacial:** GAT (2 heads).
    * **Backbone Temporal:** LSTM (batch_first=True).
    * **Pipeline:** $GNN_{t} \rightarrow Stack \rightarrow LSTM \rightarrow Linear$.

### 🧬 Configuración del Grafo y Datos
Definida en `graphCreator.py` y `main.py`:

* **Construcción del Grafo:** $k$-NN (k-Nearest Neighbors).
    * **Vecinos ($k$):** `5`.
    * **Perfil de Similitud:** `'a&g'` (Basado en vectores de Asistencia + Notas).
    * **Métrica:** Distancia Euclidiana.
* **Preprocesamiento:**
    * **Target ($Y$):** Normalizado en rango $[0, 1]$ (Nota / 10).
    * **Validación Cruzada:** 5-Fold Cross Validation (`shuffle=True`, `random_state=42`).
---

## 🛠️ Instalación y Uso

1.  **Requisitos:**
    ```bash
    pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn imageio
    ```
2.  **Estructura de Datos:**
    Asegúrate de tener los archivos CSV (`asistencia.csv`, `seguimiento.csv`, etc.) en la carpeta `./data/`.

3.  **Ejecución:**
    Para entrenar los modelos, evaluar y generar gráficas:
    ```bash
    python src/main.py
    ```

## ToDo list
* [ ] Probar otros perfiles de similitud para el grafo

---
*Trabajo de Fin de Grado - Ingeniería Informática y Matemáticas*


