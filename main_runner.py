from src.suites import Suites
from src.logger import Performance
from src.result import Result
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np


features = None
#features = np.load('./models/randomForest_Selection_50feat.npy')
#features = np.load('./models/randomForest_Selection_100feat.npy')
#features = np.load('./models/randomForest_Selection_200feat.npy')

modelSelected = 'forest'
#modelLocation = 'randomForest_noSelection50'
#modelLocation = 'randomForest_noSelection100'
modelLocation = 'randomForest_noSelection200'

#modelSelected = 'forest'
#modelLocation = 'randomForest_Selection_50'
#modelLocation = 'randomForest_Selection_100'
#modelLocation = 'randomForest_Selection_200'

#modelSelected ='annExpected'
#modelLocation = '_Drop0.5_Hidden39.0_Epoch50_Learning0.001_Size:50_LossWCategoricalCrossentropy'
#modelLocation = '_Drop0.5_Hidden39.0_Epoch50_Learning0.001_Size:100_LossWCategoricalCrossentropy'
#modelLocation = '_Drop0.5_Hidden39.0_Epoch50_Learning0.001_Size:200_LossWCategoricalCrossentropy'

#modelSelected = 'annCross'
#modelLocation = '_Drop0.5_Hidden39.0_Epoch50_Learning0.001_Size:50_Losscategorical_crossentropy'
#modelLocation = '_Drop0.5_Hidden39.0_Epoch50_Learning0.001_Size:100_Losscategorical_crossentropy'
#modelLocation = '_Drop0.5_Hidden39.0_Epoch50_Learning0.001_Size:200_Losscategorical_crossentropy'

#size = 50
#size = 100
size = 200


if modelSelected =='annExpected':
    def weightedCategoricalCrossentropy(self, y_true, y_pred):
        return K.sum(y_true * y_pred, axis=-1,keepdims=True)

    model = tf.keras.models.load_model('./models/'+modelLocation, custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})

elif modelSelected == 'annCross':
    model = tf.keras.models.load_model('./models/'+modelLocation)

elif modelSelected=='forest':
    model = pickle.load(open('./models/'+modelLocation, 'rb'))

name = modelLocation
performance = Performance()
print(name)
#name = 'nedler'
#localSearch = 'nedler'

#name = 'bfgs0.1'
#localSearch = 'bfgs0.1'

#name = 'bfgs0.3'
#localSearch = 'bfgs0.3'

#name = "Test_best_solver"
esconfig = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1] 

#name = "Test_CMA-ES"
#esconfig = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 

#name = "Test_Active_CMA-ES"
#esconfig = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#name = "Test_Elitist_CMA-ES"
#esconfig = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#name = "Test_Mirrored-pairwise_CMA-ES"
#esconfig = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]

#name = "Test_IPOP-CMA-ES"
#esconfig = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 

#name = "Test_Active_IPOP-CMA-ES"
#esconfig = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

#name = "Test_Elitist_Active_IPOP-CMA-ES"
#esconfig = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

#name = "Test_BIPOP-CMA-ES"
#esconfig = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]

#name = "Test_Active_BIPOP-CMA-ES"
#esconfig = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]

#name = "Elitist_Active_BIPOP-CMA-ES"
#esconfig = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2]


if (size is not None):
    for i in range(1,25):
        suite = Suites(instances=[6,7,8,9,10], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=esconfig, function=i, performance=performance, pflacco=True, localSearch=None)
        suite.runTestModel(ASP=model, size=size,restart=False, features= features)
        performance.saveToCSVPerformance('Test_'+name)

else:
    print("Please specify the size")