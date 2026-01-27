
from graphCreator import GraphCreator
from modelTrainer import EntrenadorGNN
from model import AdaptiveModel
from torch_geometric.data import Data
from sklearn.metrics import r2_score


def main():
    """Prueba de predicción parcial."""
    TAG = "last_flex"
    
    # Probamos a generar el grafo con datos hasta una semana concreta
    semana_objetivo = 6  # Por ejemplo, predecir la semana 6
    graph_creator = GraphCreator(semana_final=semana_objetivo)

    print("Creando grafo para predicción parcial...")
    graph = graph_creator.create_graph(cat_opt='Temp', sim_profile='a&g', k_neighbors=5, dyn_graph=True)
    print("Grafo generado:", graph)

    trainer = EntrenadorGNN()
    print("Evaluando modelo con predicción parcial...")
    stgnn, _ = trainer.load_model(f'STGNN_{TAG}',flexible=True,dir=TAG)  # Cargar un modelo preentrenado
    lstm, _ = trainer.load_model(f'LSTM_{TAG}',flexible=True,dir=TAG)

    print(f"PRUEBAS PARA EL TAG {TAG}:")

    stgnn.eval()
    pred = stgnn(graph)

    # Calcular métricas de evaluación
    y_true = graph.y.cpu().numpy().flatten()
    y_pred = pred.detach().numpy().flatten()
    r2 = r2_score(y_true, y_pred)
    print(f" STGNN: R² para predicción parcial hasta semana {semana_objetivo}: {r2:.4f}")

    lstm.eval()
    pred = lstm(graph)

    # Calcular métricas de evaluación
    y_true = graph.y.cpu().numpy().flatten()
    y_pred = pred.detach().numpy().flatten()
    r2 = r2_score(y_true, y_pred)
    print(f" LSTM: R² para predicción parcial hasta semana {semana_objetivo}: {r2:.4f}")



if __name__ == "__main__":
    main()