import pandas as pd
import openpyxl
from sklearn.cluster import MeanShift
if __name__ == "__main__":
    dataset = pd.read_csv("./data/candy.csv")
    #print(dataset.head(5))
    #X = dataset.drop('rain', axis=1)
    X = dataset.drop('competitorname', axis=1)
    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)
    dataset['meanshift'] = meanshift.labels_
    print("="*64)
    print(dataset)

    # Guardar los datos en un archivo Excel
    dataset.to_excel("Candy.xlsx", index=False)