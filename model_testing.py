from src.suites import Suites
from src.logger import Performance
from src.result import Result
from src.models import Models
import pandas as pd
import tensorflow as tf
import keras.backend as K


#depending on the configuration used in the LSTM, the model name can change. 
modelName ='_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:0_Loss_WCategoricalCrossentropy'
modelName ='_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:1_Loss_WCategoricalCrossentropy'
modelName ='_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:2_Loss_WCategoricalCrossentropy'

modelName = '_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:0_Loss_categorical_crossentropy'
modelName = '_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:1_Loss_categorical_crossentropy'
modelName = '_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:2_Loss_categorical_crossentropy'

sampleSizeValue = 200
timeSteps =2 
esconfig = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1] 

#custom loss function needs to be redefined 
def weightedCategoricalCrossentropy( y_true, y_pred):
        return K.mean(K.sum(y_true*y_pred, axis=1))

model = tf.keras.models.load_model('./models/'+modelName, custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})

performanceASP = Performance()

#test model for function 1 with dimensions 2 and 3. 
for i in range(1,2):
    suite = Suites(instances=[6,7,8,9,10], baseBudget=10000, dimensions=[2,3,5,10], esconfig=esconfig, function=i, performance=performanceASP, pflacco=True, localSearch=None)
    suite.runTestModel(ASP=model, size=sampleSizeValue ,restart=False, features=None, ASPName=modelName, stepSize=timeSteps)
    performanceASP.saveToCSVPerformance('Test_e-2'+modelName)