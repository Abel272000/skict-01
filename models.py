
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from utils import Utils

import warnings
warnings.simplefilter("ignore")
#---------GridSearchCV ---- 0.9776740115300343
class Models:
    def __init__(self):
        self.reg = {
             'LinearSVC': LinearSVC(),
             'GradientClass' : GradientBoostingClassifier(),
             'RandomForest': RandomForestClassifier(),
             'KNeighbors': KNeighborsClassifier()
        }
        self.params = {
            'LinearSVC': {
                'max_iter' : [1000],
            },
            'GradientClass' : {
                'n_estimators' : [150],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    def grid_training(self, X, y, dataset_name):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(reg, self.params[name], cv = 3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            if score > best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)
#-------------RandomizedSearchCV ---- 0.9790995282798561
class ModelsRam:
    def __init__(self):
        self.reg = {
            'LinearSVC': LinearSVC(),
            'GradientClass': GradientBoostingClassifier(),
            'RandomForest': RandomForestClassifier(),
            'KNeighbors': KNeighborsClassifier()
        }
        self.params = {
            'LinearSVC': {
                'max_iter': [1000],
            },
            'GradientClass': {
                'n_estimators': [150],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    
    def grid_training(self, X, y, dataset_name):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(random_reg.best_score_)
            if score > best_score:
                best_score = score
                best_model = random_reg.best_estimator_
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)
# 0.9897070467141725
#Score en el conjunto de prueba: 0.9916864608076009
class Modelsho:
    def __init__(self):
        self.reg = {
            'LinearSVC': LinearSVC(),
            'GradientClass': GradientBoostingClassifier(),
            'RandomForest': RandomForestClassifier(),
            'KNeighbors': KNeighborsClassifier()
        }
        self.params = {
            'LinearSVC': {
                'max_iter': [1000],
            },
            'GradientClass': {
                'n_estimators': [150],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }

    def grid_training(self, X, y, dataset_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X_train, y_train.values.ravel())
            score = np.abs(random_reg.best_score_)
            if score > best_score:
                best_score = score
                best_model = random_reg.best_estimator_
        
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)
        
        # Evaluación del mejor modelo en el conjunto de prueba
        test_score = best_model.score(X_test, y_test)
        print(f"Score en el conjunto de prueba: {test_score}")
#------------- error
class Modelslo:
    def __init__(self):
        self.reg = {
            'LinearSVC': LinearSVC(),
            'GradientClass': GradientBoostingClassifier(),
            'RandomForest': RandomForestClassifier(),
            'KNeighbors': KNeighborsClassifier(),
        }
        self.params = {
            'LinearSVC': {
                'max_iter': [1000],
            },
            'GradientClass': {
                'n_estimators': [150],
                'learning_rate': [0.01, 0.05, 0.1],
                'criterion': ['friedman_mse', 'mse']
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
        }

    def grid_training(self, X, y, dataset_name):
        loo = LeaveOneOut()
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            scores = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X_train, y_train.ravel())
                score = np.abs(random_reg.best_score_)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = random_reg.best_estimator_
        
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)
#---- si -- 0.9040576854573047
class ModelsSVRGRi:
    def __init__(self):
        self.reg = {
            'SVR' : SVR(),
            'GRADIENT' : GradientBoostingRegressor(),
            'Ridge': Ridge(),
            'RandomForest': RandomForestRegressor()
        }
        self.params = {
           'SVR' : {
               'kernel' : ['poly'], #, 'poly', 'rbf'
               'gamma' : ['scale'], #auto, 'scale'
               'C' : [5] #1,5,10
           }, 
           'GRADIENT' : {
               'loss' : ['absolute_error', 'squared_error'],
               'learning_rate' : [0.05] #0.01, 0.05, 0.1
           },
            'Ridge': {
                'alpha': [0.1],#0.1, 1.0, 10.0
                'solver': ['auto']#'auto', 'svd', 'cholesky'
            },
            'RandomForest': {
                'n_estimators': range(4,16),#[100],100, 200, 300
                'max_depth': range(2,11),#[None],, 5, 10
                'min_samples_split': [2]#2, 5, 10
            }
        }
    def grid_training(self, X,y,dataset_name ):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        print(score)
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)
#-----no vale
class ModelsRSCVRam:
    def __init__(self):
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }
        self.params = {
           'SVR': {
               'kernel': ['linear', 'poly', 'rbf'],
               'gamma': ['auto', 'scale'],
               'C': [1, 5, 10]
           },
           'GRADIENT': {
               'loss': ['absolute_error', 'squared_error'],
               'learning_rate': [0.01, 0.05, 0.1]
           }
        }
    
    def grid_training(self, X, y,dataset_name):
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(random_reg.best_score_)
            if score < best_score:
                best_score = score
                best_model = random_reg.best_estimator_
        print(score)
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)
#-----------Hold-On 0.9595065491332347
class Modelshold:
    def __init__(self):
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }
        self.params = {
           'SVR': {
               'kernel': ['rbf'],
               'gamma': ['scale'],
               'C': [5]
           },
           'GRADIENT': {
               'loss': ['absolute_error', 'squared_error'],
               'learning_rate': [0.01, 0.05, 0.1]
           }
        }
    
    def grid_training(self, X, y, dataset_name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X_train, y_train.values.ravel())
            score = np.abs(random_reg.best_score_)
            if score < best_score:
                best_score = score
                best_model = random_reg.best_estimator_
        
        print(score)
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)

#-------LOOCV --- error
class Modelsloo:
    def __init__(self):
        self.reg = {
            'SVR': SVR(),
            'GRADIENT': GradientBoostingRegressor()
        }
        self.params = {
           'SVR': {
               'kernel': ['linear'],
               'gamma': [ 'auto'],
               'C': [1]
           },
           'GRADIENT': {
               'loss': ['absolute_error'],
               'learning_rate': [0.05]
           }
        }
    
    def grid_training(self, X, y, dataset_name):
        loo = LeaveOneOut()
        
        best_score = 0
        best_model = None
        for name, reg in self.reg.items():
            scores = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                random_reg = RandomizedSearchCV(reg, self.params[name], cv=3).fit(X_train, y_train.ravel())
                score = np.abs(random_reg.best_score_)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_model = random_reg.best_estimator_
        
        print(score)
        utils = Utils()
        utils.model_export(best_model, best_score, dataset_name + name)