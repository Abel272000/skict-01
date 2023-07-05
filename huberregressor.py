import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer ##Discretizar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
RANSACRegressor, HuberRegressor
)

#Modelo de Máquinas de soporte de vectores, sub modelo La Regresión de Vectores de Soporte
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

##########################################
if __name__ == "__main__":
    dataset = pd.read_csv('./data/TDataSEnd1.csv')
    print(dataset.head(5))
    X = dataset.drop(['Calif_Revis_Seg','Toxicos'], axis=1)
    y = dataset[['Toxicos']]
    ################## Normalizar los datos ################### 

    #dataset = StandardScaler().fit_transform(dataset) #Normalizamnos los datos ##por la cantidad de los datos son muy grande las cantidades de los datos ##normaliza los campos restantes
    #print(dt_features)

    ###################Discretizar datos ###################

    # Crear el objeto KBinsDiscretizer
    #discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    # Discretizar los datos
    #dataset = discretizer.fit_transform(dataset)    
    #print(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(X,y, 
    test_size=0.3, random_state=42)
    estimadores = {
    'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),
    'RANSAC' : RANSACRegressor(),
    'HUBER' : HuberRegressor(epsilon=1.35)
    }
    warnings.simplefilter("ignore")
    for name, estimator in estimadores.items():
        #entrenamiento
        estimator.fit(X_train, y_train)
        #predicciones del conjunto de prueba
        predictions = estimator.predict(X_test)
        print("="*64)
        print(name)
        #medimos el error,datos de prueba y predicciones
        print("MSE: "+"%.10f" % float(mean_squared_error(y_test, 
        predictions)))
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real')
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.show()

