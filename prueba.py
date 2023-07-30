import joblib as jb
import numpy as np
model = jb.load('C:/Users/PERSONAL/Desktop/Aracely/models/Abeldb0.pkl')

#X_test = np.array([0,	25.6,	80.96,	21.54,	0.04,	0.63,	188.51,	2,	5,	0]) #debe dar un valor 0

X_test = np.array([0.01,23.09,89.45,21.07,0.08,	0.88,207.05,2,4,100]) #debe dar un valor 1



prediction = model.predict(X_test.reshape(1,-1))
if prediction[0] == 1:   
    print({'Su cultivo tiene la enfermedad' : list(prediction)})
else:
    print({'Su cultivo no tiene la enfermedad' : list(prediction)})