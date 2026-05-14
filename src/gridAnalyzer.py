"""
Módulo para el análisis de resultados de Grid Search.
Permite:
    - Cargar resultados desde archivos CSV.
    - Identificar las mejores configuraciones según R² y MAPE.
    - Generar gráficos detallados para visualizar el impacto de los hiperparámetros.
    - Guardar gráficos y resúmenes de las mejores configuraciones.
Uso:
    1. Ejecutar el script.
    2. Seleccionar el archivo CSV a analizar o elegir 'all' para analizar todos los archivos disponibles.
    3. Visualizar los resultados en consola y gráficos generados.
    4. Revisar los archivos guardados con los gráficos y resúmenes de las mejores configuraciones.
Nota:
    - Asegúrese de que los archivos CSV sigan el formato esperado con las columnas: 'Hidden_Dim', 'Dropout', 'LR', 'R2_Mean', 'Best_R2', 'MAPE_Mean', 'Best_MAPE'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PATH_FILES = "./data/hyperparam_opt/"

def analyze_grid_search_results(file_path, show_plots=True):
    """Analiza los resultados de un Grid Search almacenados en un archivo CSV.
    Identifica las mejores configuraciones según R² y MAPE, y genera gráficos para visualizar el impacto de los hiperparámetros.
    Args:
        file_path (str): Ruta al archivo CSV con los resultados del Grid Search.
        show_plots (bool): Indica si se deben mostrar los gráficos en pantalla.
    Returns:
        tuple: Configuraciones óptimas para R2 y MAPE.
    """
    
    df = pd.read_csv(file_path)
    
    best_r2_idx = df['R2_Mean'].idxmax()
    best_r2_config = df.loc[best_r2_idx]

    best_mape_idx = df['MAPE_Mean'].idxmin()
    best_mape_config = df.loc[best_mape_idx]

    print("=== Mejor Configuración por R2 (Maximizando) ===")
    print(best_r2_config[['Hidden_Dim', 'Dropout', 'LR', 'R2_Mean', 'MAPE_Mean']])
    print("\n=== Mejor Configuración por MAPE (Minimizando) ===")
    print(best_mape_config[['Hidden_Dim', 'Dropout', 'LR', 'R2_Mean', 'MAPE_Mean']])

    
    sns.set(style="whitegrid")

    g_r2 = sns.catplot(
        data=df, 
        x='Hidden_Dim', 
        y='R2_Mean', 
        hue='LR', 
        col='Dropout', 
        kind='point', 
        palette='Dark2', 
        height=4, 
        aspect=1
    )
    g_r2.fig.subplots_adjust(top=0.85)
    g_r2.fig.suptitle('R2 Mean: Impacto de Hidden Dim, LR y Dropout')


    g_mape = sns.catplot(
        data=df, 
        x='Hidden_Dim', 
        y='MAPE_Mean', 
        hue='LR', 
        col='Dropout', 
        kind='point', 
        palette='magma_r',
        height=4, 
        aspect=1
    )
    g_mape.figure.subplots_adjust(top=0.85)
    g_mape.figure.suptitle('MAPE Mean: Impacto de Hidden Dim, LR y Dropout')

    
    path = file_path.replace(".csv", "_R2_plot.png")
    g_r2.savefig(path)
    print(f"📁 Gráfico de R2 guardado en '{path}'")

    path = file_path.replace(".csv", "_MAPE_plot.png")
    g_mape.savefig(path)
    print(f"📁 Gráfico de MAPE guardado en '{path}'")
    if show_plots:
        plt.show()
    
    return best_r2_config, best_mape_config

if __name__ == "__main__":
    """
    Ejecuta el análisis de los resultados del Grid Search.
    Permite seleccionar un archivo específico o analizar todos los archivos disponibles en la carpeta.
    """
    
    print("Archivos disponibles para análisis:")
    archivos = os.listdir(PATH_FILES)
    archivos = [f for f in archivos if f.endswith(".csv") and "grid_search_results" in f]
    for i,file in enumerate(archivos):
        print(f"\t{i}: {file}")
    

    idx = input("\nIngrese el número del archivo que desea analizar: ")
    if idx == 'all':    # Analizar todos los archivos disponibles y guardar un resumen de las mejores configuraciones
        print("Análisis de todos los archivos:")    
        results_r2 = []
        results_mape = []
        for file in archivos:
            file_path = os.path.join(PATH_FILES, file)
            print(f"\nAnalizando el archivo: {file_path}\n")
            best_r2_config, best_mape_config = analyze_grid_search_results(file_path, show_plots=False)
            results_r2.append({
                'file': file,
                'Hidden_Dim': best_r2_config['Hidden_Dim'],
                'Dropout': best_r2_config['Dropout'],
                'LR': best_r2_config['LR'],
                'R2_Mean': best_r2_config['R2_Mean'],
                'Best_R2': best_r2_config['Best_R2'],
                'MAPE_Mean': best_r2_config['MAPE_Mean'],
                'Best_MAPE': best_r2_config['Best_MAPE']
            })
            results_mape.append({
                'file': file,
                'Hidden_Dim': best_mape_config['Hidden_Dim'],
                'Dropout': best_mape_config['Dropout'],
                'LR': best_mape_config['LR'],
                'R2_Mean': best_mape_config['R2_Mean'],
                'Best_R2': best_mape_config['Best_R2'],
                'MAPE_Mean': best_mape_config['MAPE_Mean'],
                'Best_MAPE': best_mape_config['Best_MAPE']
            })
        
        # Guardamos en un CSV resumen de los mejores resultados
        df_summary_r2 = pd.DataFrame(results_r2)
        df_summary_mape = pd.DataFrame(results_mape)
        summary_path = os.path.join(PATH_FILES, "summary_best_configs.csv")
        df_summary_r2.to_csv(summary_path.replace(".csv", "_R2.csv"), index=False)
        df_summary_mape.to_csv(summary_path.replace(".csv", "_MAPE.csv"), index=False)
    elif 0 <= int(idx) < len(archivos):
        file_path = os.path.join(PATH_FILES, archivos[int(idx)])
        print(f"\nAnalizando el archivo: {file_path}\n")
        analyze_grid_search_results(file_path)
    else:
        print("Número inválido. Por favor, ingrese un número válido o 'all' para analizar todos los archivos.")