
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt


if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    dataset.drop(["competitorname"], axis=1, inplace=True)
    # Aplicar el clustering jerárquico
    linkage_matrix = linkage(dataset, method='ward', metric='euclidean')
    
    # Plotear el dendrograma
    plt.figure(figsize=(8, 5)) #crea una nueva figura de tamaño 10x6 pulgadas para visualizar el 
    #dendrograma.
    dendrogram(linkage_matrix, truncate_mode='level', p=3) # Esta línea genera el dendrograma utilizando la 
    #matriz de enlace (linkage_matrix) calculada previamente mediante el clustering jerárquico. 
    #El parámetro truncate_mode se establece en 'level', lo que significa que se mostrarán todas las fusiones 
    # de clusters hasta el nivel especificado en el parámetro p
    plt.xlabel('muestras') #Indica las muestras o puntos de datos que se están agrupando.
    plt.ylabel('distancia')#este eje muestra la distancia o similitud para fusionar los clusters.
    plt.title('Clustering jerarquico para mi nena hermosa')
    plt.show()