import pandas as pd
from sklearn.neighbors import KNeighborsClassifier ##este clacificador
from sklearn.ensemble import BaggingClassifier ##el mismo clasificador
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler #Normalizar los datos ## libreria para normalizar
from sklearn.preprocessing import KBinsDiscretizer ##Discretizar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/dataOT.csv')
    #print(dt_heart['target'].describe()
    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']

    ################### Normalizar los datos ################### 

    #dt_heart = StandardScaler().fit_transform(dt_heart) #Normalizamnos los datos ##por la cantidad de los datos son muy grande las cantidades de los datos ##normaliza los campos restantes
    #print(dt_features)

    ###################Discretizar datos ###################

    # Crear el objeto KBinsDiscretizer
    #discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # Discretizar los datos
    #dt_heart = discretizer.fit_transform(dt_heart)    
    #print(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size=0.35, random_state=1)
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_prediction = knn_class.predict(X_test)
    print('='*64)
    print('SCORE con KNN: ', accuracy_score(knn_prediction, y_test))
    
    
    ##bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), 
    #n_estimators=50).fit(X_train, y_train) # base_estimator pide el estimador en el que va a estar basado nuestro metodo || n_estimators nos pide cuantos de estos modelos vamos a utilizar
    ##bag_pred = bag_class.predict(X_test)
    #print('='*64)
    #print(accuracy_score(bag_pred, y_test))

    ##estimadores comparativa
    estimators = {
    'LogisticRegression' : LogisticRegression(),
    'SVC' : SVC(),
    'LinearSVC' : LinearSVC(),
    'SGD' : SGDClassifier(loss="hinge", penalty="l2", 
    max_iter=5),
    'KNN' : KNeighborsClassifier(),
    'DecisionTreeClf' : DecisionTreeClassifier(),
    'RandomTreeForest' : RandomForestClassifier(random_state=0)
    }
    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, 
        n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_class.predict(X_test)
        print('='*64)
        print('SCORE Bagging with {} : {}'.format(name, 
        accuracy_score(bag_predict, y_test)))