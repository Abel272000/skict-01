from utils import Utils
from models import Models
from models import ModelsRam
from models import ModelsRSCVRam
from models import ModelsSVRGRi
from models import Modelsho
from models import Modelslo
from models import Modelsloo
from models import Modelshold
#from GridSearchCV import Models

import warnings
warnings.simplefilter("ignore")

utils = Utils()
models = Modelsloo()
ds = utils.load_from_csv('data/dataOT.csv')
print(ds)
X, y = utils.features_target(ds, ['INCIDENCIA'],['INCIDENCIA'])

models.grid_training(X,y, 'dataOT')