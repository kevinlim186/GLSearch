from src.suites import Suites
from src.logger import Performance
from src.result import Result
import pandas as pd
import tensorflow as tf


def weightedCategoricalCrossentropy(self, y_true, y_pred):
    return K.sum(y_true * y_pred, axis=-1,keepdims=True)

model = tf.keras.models.load_model('./models/_Drop0.5_Hidden0.75_Epoch2000_Learning0.001_Dataset:50', custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})

name = 'Drop0.5_Hidden0.75_Epoch2000_Learning0.001_Dataset:50_restart_false'

performance = Performance()
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


'''
#function 6 to 7
for i in range(6,8):
    suite = Suites(instances=[1,2,3,4,5], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], function=i, performance=performance, pflacco=False)

    suite.runDataGathering()
'''

'''
for i in range(1,25):
    suite = Suites(instances=[6,7,8,9,10], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=esconfig, function=i, performance=performance, pflacco=False, localSearch=localSearch)
    suite.runTestSuite()

performance.saveToCSVPerformance('Test_'+name)
'''

for i in range(1,25):
    suite = Suites(instances=[6,7,8,9,10], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=esconfig, function=i, performance=performance, pflacco=True, localSearch=None)
    suite.runTestModel(ASP=model, size=50,restart=False)
    performance.saveToCSVPerformance('Test_'+name)

