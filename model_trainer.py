#import autosklearn.regression
import pandas as pd
import numpy as np
import pickle
from src.result import Result
from src.models import Models

#dateset
dataset = '50'
#dataset = '100'
#dataset = '200'


#Interface
interface = None
interface = ['ela meta.lin simple.adj r2', 'nbc.nb fitness.cor', 'ela meta.lin simple.intercept', 'dim', 'ela meta.quad w interact.adj r2', 'nbc.nn nb.sd ratio', 'ela meta.lin w interact.adj r2', 'ela meta.quad simple.adj r2', 'budget.used']

#define the model
#modelSelected='ANNCross0.75'
#modelSelected='ANNExpected0.75'
#modelSelected='ANNCross125'
#modelSelected='ANNExpected125'
#modelSelected='Forest'
modelSelected='ForestFeature'
#modelSelected = 'LSTM'

print(dataset+ ' for ' +  modelSelected)


result = Result()


#load performance
perf1 = pd.read_csv("./perf/all_performance.csv")

#Load ELA Files
if dataset == '50':
    ela50 = pd.read_csv("./perf/train50.csv").rename(columns={'name':'oldname'})
    ela50['name'] = ela50['oldname'] + '_ela_sample_50'
    ela50 = ela50.iloc[:,23:]
    traindata= ela50
elif dataset == '100':
    ela100 = pd.read_csv("./perf/train100.csv").rename(columns={'name':'oldname'})
    ela100['name'] = ela100['oldname'] + '_ela_sample_100'
    ela100 = ela100.iloc[:,23:]
    traindata= ela100
elif dataset == '200':
    ela200 = pd.read_csv("./perf/train200.csv").rename(columns={'name':'oldname'})
    ela200['name'] = ela200['oldname'] + '_ela_sample_200'
    ela200 = ela200.iloc[:,23:]
    traindata= ela200

    
result.addPerformance(perf1)
result.addELA(traindata)

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
elif modelSelected =='LSTM':
    model = Models(Xtrain,Ytrain,_shuffle=False)
    model.trainLSTM(size=dataset)


