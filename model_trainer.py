#import autosklearn.regression
import pandas as pd
import numpy as np
import pickle
from src.result import Result
from src.models import Models

#dateset
#dataset = '0' #50D
#dataset = '1' #100D
dataset = '2' #200D
#dataset = '3' #5G
#dataset = '4' #50G
#dataset = '5' #100G



#Interface
interface = None
#interface = ['ela_meta.lin_simple.adj_r2', 'nbc.nb_fitness.cor', 'ela_meta.lin_simple.intercept', 'basic.dim', 'ela_meta.quad_w_interact.adj_r2', 'nbc.nn_nb.sd_ratio', 'ela_meta.lin_w_interact.adj_r2', 'ela_meta.quad_simple.adj_r2', 'budget.used']

#define the model
#modelSelected='ANNCross0.75'
#modelSelected='ANNExpected0.75'
#modelSelected='ANNCross125'
#modelSelected='ANNExpected125'
#modelSelected='Forest'
#modelSelected='ForestFeature'
modelSelected = 'LSTM'
stepSize =1

#loss = 'categorical_crossentropy'
loss = 'WCategoricalCrossentropy'

print(dataset+ ' for ' +  modelSelected)


result = Result()


#load performance
perf = pd.read_csv("./perf/Performance_Gathering.csv")

#Load ELA Files
traindata= pd.read_csv("./perf/Performance_Gathering_ELA.csv")

result.addPerformance(perf)
result.addELA(traindata)


if modelSelected =='LSTM':
    Xtrain, Ytrain = result.createTrainSet(dataset=dataset, algorithm=None, reset=False, interface=interface, RNN=stepSize)
    model = Models(Xtrain,Ytrain,_shuffle=False)
    model.trainLSTM(stepSize=stepSize, size=dataset, loss=loss)

else:
    Xtrain, Ytrain = result.createTrainSet(dataset=dataset, algorithm=None, reset=False, interface=interface)

    if modelSelected =='ANNCross0.75':
        model = Models(Xtrain,Ytrain,_shuffle=True)
        model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*0.75, epoch=50, size=dataset,learning=0.001, output_size=len(Ytrain[0]), loss='categorical_crossentropy')
    elif modelSelected =='ANNCross125':
        model = Models(Xtrain,Ytrain,_shuffle=True)
        model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*2+1, epoch=50, size=dataset,learning=0.001, output_size=len(Ytrain[0]), loss='categorical_crossentropy')
    elif modelSelected =='ANNExpected0.75':
        model = Models(Xtrain,Ytrain,_shuffle=True)
        model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*0.75, epoch=50, size=dataset,learning=0.001, output_size=len(Ytrain[0]), loss='WCategoricalCrossentropy')
    elif modelSelected =='ANNExpected125':
        model = Models(Xtrain,Ytrain,_shuffle=True)
        model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*2+1, epoch=50, size=dataset,learning=0.001, output_size=len(Ytrain[0]), loss='WCategoricalCrossentropy')
    elif modelSelected =='Forest':
        model = Models(Xtrain,Ytrain,_shuffle=True)
        model.trainRandomForest(size=dataset, selection=False)
    elif modelSelected == 'ForestFeature':
        model = Models(Xtrain,Ytrain,_shuffle=True)
        model.trainRandomForest(size=dataset, selection=True)


