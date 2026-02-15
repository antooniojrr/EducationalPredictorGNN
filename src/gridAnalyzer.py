import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PATH_FILES = "./data/hyperparam_opt/"

def analyze_grid_search_results(file_path, show_plots=True):

    # 1. Cargar los datos
    df = pd.read_csv(file_path)
    # 2. Análisis: Encontrar la mejor configuración
    # Mejor R2 (Mayor es mejor)
    best_r2_idx = df['R2_Mean'].idxmax()
    best_r2_config = df.loc[best_r2_idx]

    # Mejor MAPE (Menor es mejor)
    best_mape_idx = df['MAPE_Mean'].idxmin()
    best_mape_config = df.loc[best_mape_idx]

    print("=== Mejor Configuración por R2 (Maximizando) ===")
    print(best_r2_config[['Hidden_Dim', 'Dropout', 'LR', 'R2_Mean', 'MAPE_Mean']])
    print("\n=== Mejor Configuración por MAPE (Minimizando) ===")
    print(best_mape_config[['Hidden_Dim', 'Dropout', 'LR', 'R2_Mean', 'MAPE_Mean']])

    # 3. Visualización
    sns.set(style="whitegrid")

    # Crear gráficos detallados separando por Dropout para ver el efecto de Hidden_Dim y LR
    # Gráfico para R2
    g_r2 = sns.catplot(
        data=df, 
        x='Hidden_Dim', 
        y='R2_Mean', 
        hue='LR', 
        col='Dropout', 
        kind='point', 
        palette='viridis', 
        height=4, 
        aspect=1
    )
    g_r2.fig.subplots_adjust(top=0.85)
    g_r2.fig.suptitle('R2 Mean: Impacto de Hidden Dim, LR y Dropout')

    # Gráfico para MAPE
    g_mape = sns.catplot(
        data=df, 
        x='Hidden_Dim', 
        y='MAPE_Mean', 
        hue='LR', 
        col='Dropout', 
        kind='point', 
        palette='magma_r', # Invertido porque menor es mejor
        height=4, 
        aspect=1
    )
    g_mape.figure.subplots_adjust(top=0.85)
    g_mape.figure.suptitle('MAPE Mean: Impacto de Hidden Dim, LR y Dropout')

    # Guardamos los gráficos
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

    # Muestro los archivos disponibles
    print("Archivos disponibles para análisis:")
    archivos = os.listdir(PATH_FILES)
    archivos = [f for f in archivos if f.endswith(".csv") and "grid_search_results" in f]
    for i,file in enumerate(archivos):
        print(f"\t{i}: {file}")
    
    # Selecciono el archivo a analizar
    idx = input("\nIngrese el número del archivo que desea analizar: ")
    if idx == 'all':
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