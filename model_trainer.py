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

#define the model
#modelSelected='ANNCross0.75'
modelSelected='ANNExpected0.75'
#modelSelected='ANNCross125'
#modelSelected='ANNExpected125'
#modelSelected='Forest'
#modelSelected='ForestFeature'




#Load ELA Files
result = Result()
ela50 = pd.read_csv("./perf/train50.csv").rename(columns={'name':'oldname'})
ela100 = pd.read_csv("./perf/train100.csv").rename(columns={'name':'oldname'})
ela200 = pd.read_csv("./perf/train200.csv").rename(columns={'name':'oldname'})

ela50['name'] = ela50['oldname'] + '_ela_sample_50'
ela50 = ela50.iloc[:,23:]

ela100['name'] = ela100['oldname'] + '_ela_sample_100'
ela100 = ela100.iloc[:,23:]

ela200['name'] = ela200['oldname'] + '_ela_sample_200'
ela200 = ela200.iloc[:,23:]

#load performance
perf1 = pd.read_csv("./perf/all_performance.csv")


if dataset == '50':
    traindata= ela50
elif dataset == '100':
    traindata= ela100
elif dataset == '200':
    traindata= ela200

    
result.addPerformance(perf1)
result.addELA(traindata)

Xtrain, Ytest = result.createTrainSet(dataset=dataset, algorithm=None, reset=False)

model = Models(Xtrain,Ytest)
if modelSelected =='ANNCross0.75':
    model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*0.75, epoch=2000, size=dataset,learning=0.001, output_size=len(Ytest[0]), loss='categorical_crossentropy')
elif modelSelected =='ANNCross125':
    model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*2+1, epoch=2000, size=dataset,learning=0.001, output_size=len(Ytest[0]), loss='categorical_crossentropy')
elif modelSelected =='ANNExpected0.75':
    model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*0.75, epoch=2000, size=dataset,learning=0.001, output_size=len(Ytest[0]), loss='WCategoricalCrossentropy')
elif modelSelected =='ANNExpected125':
    model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=len(Xtrain[0])*2+1, epoch=2000, size=dataset,learning=0.001, output_size=len(Ytest[0]), loss='WCategoricalCrossentropy')
elif modelSelected =='Forest':
    model.trainRandomForest(size=dataset, selection=False)
elif modelSelected == 'ForestFeature':
    model.trainRandomForest(size=dataset, selection=True)


