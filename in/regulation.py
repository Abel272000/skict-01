# Importamos las bibliotecas
import pandas as pd
import sklearn

# Importamos los modelos de sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler #normalizacion
from sklearn.preprocessing import KBinsDiscretizer #discretizacion
# Importamos las metricas de entrenamiento y el error medio cuadrado
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #error medio cuadrado

#####################################
if __name__ == "__main__":
    # Importamos el dataset del 2017 
    dataset = pd.read_csv('./data/dataOT.csv')
    # Mostramos el reporte estadistico
    ##print(dataset.describe())
    # Vamos a elegir los features que vamos a usar //gdp economia
    X = dataset[['Rain', 'Temperature', 'RH', 'WindSpeed', 'WindDirection', 'PLANTA', 'FRUTO', 'SEVERIDAD']]
    # Definimos nuestro objetivo, que sera nuestro data set, pero solo en la columna score 
    y = dataset[['INCIDENCIA']]
    
    #NORMALIZACION
    #dataset = StandardScaler().fit_transform(dataset)

    #DISCRETIZACION
    # Crear el objeto KBinsDiscretizer
    #discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # Discretizar los datos
    #dataset = discretizer.fit_transform(dataset)    
    #print(dt_features)

    ##datos totales 
    # Imprimimos los conjutos que creamos
    # En nuestros features tendremos definidos 155 registros, uno por cada pais, 7 colunas 1 por cada pais 
    #print(X.shape)
    # Y 155 para nuestra columna para nuestro target 
    #print(y.shape)

    
    # Aquí vamos a partir nuestro entrenaminto en training y test, no hay olvidar el orden
    # Con el test size elejimos nuestro porcetaje de datos para training 
    ##25% de utilizacion de datos
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
    ##metodos de regularizacion
    # Aquí definimos nuestros regresores uno por 1 y llamamos el fit o ajuste 
    modelLinear = LinearRegression().fit(X_train, y_train) ##datos de entrenamiento y prueba 
    # Vamos calcular la prediccion que nos bota con la funcion predict con la regresion lineal 
    # y le vamos a mandar el test 
    y_predict_linear = modelLinear.predict(X_test)
    ##metodos de regularizacion
    
    # Configuramos alpha, que es valor labda y entre mas valor tenga alpha en lasso mas penalizacion 
    # vamos a tener y lo entrenamos con la función fit 
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    # Hacemos una prediccion para ver si es mejor o peor de lo que teniamos en el modelo lineal sobre
    # exactamente los mismos datos que teníamos anteriormente 
    y_predict_lasso = modelLasso.predict(X_test)
    # Hacemos la misma predicción, pero para nuestra regresion ridge ##defaul =1 
    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    # Calculamos el valor predicho para nuestra regresión ridge 
    y_predict_ridge = modelRidge.predict(X_test)
    # Hacemos la misma predicción, pero para nuestra regresion ElasticNet 
    modelElasticNet = ElasticNet(random_state=0).fit(X_train, y_train)
    # Calculamos el valor predicho para nuestra regresión ElasticNet 
    y_pred_elastic = modelElasticNet.predict(X_test)

    ##calcular la perdida y saber cual me da el menor valor donde ese modelo  va hacer el mejor 
    # Calculamos la perdida para cada uno de los modelos que entrenamos, empezaremos con nuestro modelo 
    # lineal, con el error medio cuadratico y lo vamos a aplicar con los datos de prueba con la prediccion 
    # que hicimos 
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    # Mostramos la perdida lineal con la variable que acabamos de calcular
    ##sin aplicar regularizacion
    print( "Linear Loss. "+"%.10f" % float(linear_loss))
    # Mostramos nuestra perdida Lasso, con la variable lasso loss 
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss. "+"%.10f" % float( lasso_loss))
    # Mostramos nuestra perdida de Ridge con la variable Ridge loss 
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss: "+"%.10f" % float(ridge_loss))
    # Mostramos nuestra perdida de ElasticNet con la variable Elastic loss
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)
    print("ElasticNet Loss: "+"%.10f" % float(elastic_loss))



    # Imprimimos las coficientes para ver como afecta a cada una de las regresiones 
    # La lines "="*32 lo unico que hara es repetirme si simbolo de igual 32 veces 
    print("="*32)
    print("Coeficientes linear: ")
    # Esta informacion la podemos encontrar en la variable coef_ 
    print(modelLinear.coef_)
    print("="*32)
    ##MAYOR PESO
    print("Coeficientes lasso: ")
    # Esta informacion la podemos encontrar en la variable coef_ 
    print(modelLasso.coef_)
    # Hacemos lo mismo con ridge 
    print("="*32)
    print("Coeficientes ridge:")
    print(modelRidge.coef_) 
    # Hacemos lo mismo con elastic 
    print("="*32)
    print("Coeficientes elastic net:")
    print(modelElasticNet.coef_) 
    #Calculamos nuestra exactitud de nuestra predicción lineal
    print("="*32)
    print("Score Lineal",modelLinear.score(X_test,y_test))
    #Calculamos nuestra exactitud de nuestra predicción Lasso
    print("="*32)
    print("Score Lasso",modelLasso.score(X_test,y_test))
    #Calculamos nuestra exactitud de nuestra predicción Ridge
    print("="*32)
    print("Score Ridge",modelRidge.score(X_test,y_test))
    #Calculamos nuestra exactitud de nuestra predicción Elastic Net
    print("="*32)
    print("Score ElasticNet",modelElasticNet.score(X_test,y_test))
##interpretar bien los datos