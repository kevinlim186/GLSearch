from src.suites import Suites
from src.logger import Performance
from src.result import Result
from src.models import Models
import pandas as pd
import tensorflow as tf
import keras.backend as K

####################### Data Gathering #############################
esconfig = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1] 
performance = Performance()


#sample run gathering data for function 1, instances 1 and 2 with dimensions 2 and 3. 
for i in range(1,2):
    suite = Suites(instances=[1,2], baseBudget=10000, dimensions=[2,3], esconfig=esconfig, function=i, performance=performance, pflacco=True, localSearch=None)
    suite.runDataGathering()
    performance.saveToCSVPerformance('DataGathering')
    performance.saveToCSVELA('DataGathering')


####################### Data Preprocessing #############################
sampleSize = '2'  # choose between 0 to 2. 0 correspondes to 50D, 1 to 100D, 2 to 200D. 3 to 5 is also available but for generation based scaling
timeSteps = 2  # the number of time steps for the LSTM network


result = Result()

#load the performance and ELA files generated from data gathering
performance = pd.read_csv("./perf/DataGathering_performance.csv")
ela = pd.read_csv("./perf/DataGathering_elaFeatures.csv")

result.addPerformance(performance)
result.addELA(ela)

#this could sometimes fail if the training sample does not contain at least two time steps. This could happen if the CMA-ES finds the optimal value before the 2nd checkpoint
Xtrain, Ytrain = result.createTrainSet(dataset=sampleSize, algorithm=None, reset=False, interface=None, RNN=timeSteps)


####################### Model Training #############################
loss='WCategoricalCrossentropy' # loss function chosen. choose between "WCategoricalCrossentropy" or "categorical_crossentropy". WCategoricalCrossentropy is the expected loss described in the thesis.

model = Models(Xtrain,Ytrain,_shuffle=False)
model.trainLSTM(stepSize=timeSteps, size=sampleSize, loss=loss)


####################### Model Testing #############################

#depending on the configuration used in the LSTM, the model name can change. 
modelName ='_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:2_Loss_WCategoricalCrossentropy'
sampleSizeValue = 200


#custom loss function needs to be redefined 
def weightedCategoricalCrossentropy(self, y_true, y_pred):
        return K.mean(K.sum(y_true*y_pred, axis=1))

model = tf.keras.models.load_model('./models/'+modelName, custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})

performanceASP = Performance()

#test model for function 1, instances 6 and 7 with dimensions 2 and 3. 
for i in range(1,2):
    suite = Suites(instances=[6,7], baseBudget=10000, dimensions=[2,3], esconfig=esconfig, function=i, performance=performanceASP, pflacco=True, localSearch=None)
    suite.runTestModel(ASP=model, size=sampleSizeValue ,restart=False, features=None, ASPName=modelName, stepSize=timeSteps)
    performanceASP.saveToCSVPerformance('Test_'+modelName)


####################### Performance Reporting #############################
performanceBenchmark = Performance()
#create a benchmark for function 1, instances 6 and 7 with dimensions 2 and 3. 
for i in range(1,2):
    suite = Suites(instances=[6,7], baseBudget=10000, dimensions=[2,3], esconfig=esconfig, function=i, performance=performanceBenchmark , pflacco=True, localSearch=None)
    suite.runDataGathering()
    performanceBenchmark.saveToCSVPerformance('Benchmark')


#This function is needed to properly aggregate the performance. The algorithm chosen is included in the name so this needs to be removed.
def removeAlgorithmChosen(dataframe):
    dataframe['name'] = dataframe['name'].str.replace('::Base','')
    dataframe['name'] = dataframe['name'].str.replace(':Base','')
    dataframe['name'] = dataframe['name'].str.replace(':bfgs','')
    dataframe['name'] = dataframe['name'].str.replace(':nelder','')
    dataframe['name'] = dataframe['name'].str.replace('Local:','Local')


#Load the files
benchmark = pd.read_csv("./perf/Benchmark_performance.csv")
ASPPerformance = pd.read_csv("./perf/Test__RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:2_Loss_WCategoricalCrossentropy_performance.csv")

removeAlgorithmChosen(ASPPerformance)

result= Result()
result.addPerformance(ASPPerformance, benchmark)

performance, _ = result.calculatePerformance()

#access the performance information in the performance.csv file
performance.to_csv('performance.csv')