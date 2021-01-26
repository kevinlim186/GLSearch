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


sampleSize = '2'  # choose between 0 to 2. 0 correspondes to 50D, 1 to 100D, 2 to 200D. 3 to 5 is also available but for generation based scaling
timeSteps = 2  # the number of time steps for the LSTM network
precision_value = 'ert-2'

#set to precision value to negative 2
result = Result(ert_column=precision_value)

#load the performance and ELA files generated from data gathering
performance = pd.read_csv("./perf/DataGathering_performance.csv")
ela = pd.read_csv("./perf/DataGathering_elaFeatures.csv")

result.addPerformance(performance)
result.addELA(ela)

#this could sometimes fail if the training sample does not contain at least two time steps. This could happen if the CMA-ES finds the optimal value before the 2nd checkpoint
Xtrain, Ytrain = result.createTrainSet(dataset=sampleSize, algorithm=None, reset=False, interface=None, RNN=timeSteps)

loss='WCategoricalCrossentropy' # loss function chosen. choose between "WCategoricalCrossentropy" or "categorical_crossentropy". WCategoricalCrossentropy is the expected loss described in the thesis.

model = Models(Xtrain,Ytrain,_shuffle=False)
model.trainLSTM(stepSize=timeSteps, size=sampleSize, loss=loss, precision_value=precision_value)