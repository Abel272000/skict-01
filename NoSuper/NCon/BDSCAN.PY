import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
    # Cargar el dataset
    dataset = pd.read_csv("./data/candy.csv")
    
    # Extraer las características del dataset
    features = dataset.drop(["competitorname"], axis=1)
    features = dataset.iloc[:, 1:].values
    
    # Aplicar el algoritmo DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)#eps (epsilon) Especifica la 
    #distancia máxima entre dos puntos para que sean considerados vecinos.
    #min_samples Especifica el número mínimo de puntos dentro de un vecindario para 
    # que un punto se considere como núcleo (core point)
    dbscan.fit(features)
    
    # Obtener las etiquetas de los clusters asignados a cada punto
    labels = dbscan.labels_
    
    # Obtener el número de clusters encontrados (excluyendo el ruido)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Mostrar los resultados
    print(f"Número de clusters encontrados: {n_clusters}")
    print("Etiquetas de los clusters:")
    print(labels)
    plt.scatter(features[:, :], features[:, :])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering')
    plt.show()