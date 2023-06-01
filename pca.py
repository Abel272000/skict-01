import pandas as pd #importamos pandas ##analisis de datos alias pd ##manipulacion de datos #se puede realizar lecturas de datos
import sklearn #biblioteca de aprendizaje automático ##paquete varias librerias
import matplotlib.pyplot as plt #Librería especializada en la creación de gráficos 
from sklearn.decomposition import PCA #importamos algorimo PCA ##
from sklearn.decomposition import IncrementalPCA #importamos algorimo PCA 
from sklearn.linear_model import LogisticRegression #clasificación y análisis predictivo ## algoritmo para poder realizar dicha proyeccion ## calculo predictivo
from sklearn.preprocessing import StandardScaler #Normalizar los datos ## libreria para normalizar
#supervisados datos etiquetados
from sklearn.model_selection import train_test_split #permite hacer una división de un conjunto de datos en dos ##partir datos 1 para entrenamiento 2 prueba
#varios scrip
#bloques de entrenamiento y prueba de un modelo
if __name__ == '__main__':
    dt_data=pd.read_csv('./data/dataO1.csv') #en el directorio el punto, ubicacion, envio de datos

    #print(dt_data.head(5)) #imprimimos los 5 primeros datos

    ##10 datos 9 datos 1 etiquetado

    dt_features=dt_data.drop(['INCIDENCIA'],axis=1) #las featurus sin el target ##solo necesito 9 datos
    dt_incidecia = dt_data['INCIDENCIA'] #obtenemos el target #separamos y obtenemos dos conjuntos 
    
    dt_features = StandardScaler().fit_transform(dt_features) #Normalizamnos los datos ##por la cantidad de los datos son muy grande las cantidades de los datos ##normaliza los campos restantes

    ##entrenamiento y se envia los Feature y la clase predictiva mas el tamaño de los 10 conjuntos solo 30 voy a ocupar para entrenamiento y prueba el 0.30% del total de datos
    ##random_state=42 porque el numero 42 
    ##separacion
    X_train,X_test,y_train,y_test =train_test_split(dt_features,dt_incidecia,test_size=0.30,random_state=42)
    print(X_train.shape) #consultar la forma de la tabla con pandas
    print(y_train.shape) ##total de datos (717,) total de columnas (,13)
    
    ##algoritmo PCA
    pca=PCA(n_components=3)##de las 13 columnas solo ocupamos 3 columnas artificiales..?? variables artificiales
    # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    ##enviamos todos los datos
    pca.fit(X_train)
    
    ##
    #Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.
    '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
    poco a poco y combinarlos en el resultado final'''
    ipca=IncrementalPCA(n_components=3,batch_size=10) #tamaño de bloques, no manda a entrear todos los datos
    ##batch envia pocos datos en este caso envia bloques de 10 en 10
    #Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
    ipca.fit(X_train)

    ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
    me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
    en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
    para nuestro modelo '''
    
    ##la lonjitud desde 0 hasta tal valor 
    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_) #gneera desde 0 hasta los componentes
    #plt.show() #---grafica----
    ##grafico 0.20% relacion 
    
    ##

    ##algoritmo de regrecion
    #Ahora vamos a configurar nuestra regresión logística
    logistic=LogisticRegression(solver='lbfgs') ##predetermindado
    
    ##
    
    # Configuramos los datos de entrenamiento
    dt_train = pca.transform(X_train)#conjunto de entrenamiento 
    dt_test = pca.transform(X_test)#conjunto de prueba
    
    ##
    
    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train) #mandasmos a regresion logistica los dos datasets
    
    ##
    
    #Calculamos nuestra exactitud de nuestra predicción
    ##
    print("SCORE PCA: ", logistic.score(dt_test, y_test))##internamente imprimo la prediccion y se envia los datos de entremaiento
    
    ##
    
    #Configuramos los datos de entrenamiento
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)

    ##

    # Mandamos los data frames la la regresión logística
    logistic.fit(dt_train, y_train)

    ##

    #Calculamos nuestra exactitud de nuestra predicción
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))

    