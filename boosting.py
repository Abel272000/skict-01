import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer ##Discretizar
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/TDataSEnd1.csv')
    #print(dt_heart['target'].describe())
    x = dt_heart.drop(['Toxicos'], axis=1)
    y = dt_heart['Toxicos']
    
    #normalizacion
    #dt_heart = StandardScaler().fit_transform(dt_heart)
    ###################Discretizar datos ###################

    # Crear el objeto KBinsDiscretizer
    #discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    # Discretizar los datos
    #dt_heart = discretizer.fit_transform(dt_heart)    
    #print(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(x, y, 
    test_size=0.35, random_state=1)
    '''boosting = 
    GradientBoostingClassifier(loss='exponential',learning_rate=0.15, 
    n_estimators=188, max_depth=5).fit(X_train, y_train)
    boosting_pred=boosting.predict(X_test)
    print('='*64)
    print(accuracy_score(boosting_pred, y_test))'''
    #obtenemos el mejor resultado junto con el estimador
    estimators = range(2, 100, 2)
    total_accuracy = []
    best_result = {'result' : 0, 'n_estimator': 1}
    for i in estimators:
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)
        new_accuracy = accuracy_score(boost_pred, y_test)
        total_accuracy.append(new_accuracy)
        if new_accuracy > best_result['result']:
            best_result['result'] = new_accuracy
            best_result['n_estimator'] = i
            print(best_result)


    