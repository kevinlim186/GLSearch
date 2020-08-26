#import autosklearn.regression
import pandas as pd
import numpy as np
import pickle
from src.result import Result
from src.models import Models


result = Result()
#dateset
dataset = '50'
#dataset = '100'
#dataset = '200'

#Load ELA Files
ela123 = pd.read_csv("./GLSearch/perf/Function_1_2_3_elaFeatures.csv")
ela456 = pd.read_csv("./GLSearch/perf/Function_4_5_6_elaFeatures.csv")
ela789 = pd.read_csv("./GLSearch/perf/Function_7_8_9_elaFeatures.csv")
ela101112 = pd.read_csv("./GLSearch/perf/Function_10_11_12_elaFeatures.csv")
ela131415 = pd.read_csv("./GLSearch/perf/Function_13_14_15_elaFeatures.csv")
ela161718 = pd.read_csv("./GLSearch/perf/Function_16_17_18_elaFeatures.csv")
ela1617181 = pd.read_csv("./GLSearch/perf/Function_16_17_18_elaFeatures_1.csv")
ela192021 = pd.read_csv("./GLSearch/perf/Function_19_20_21_elaFeatures.csv")
ela1920211 = pd.read_csv("./GLSearch/perf/Function_19_20_21_elaFeatures_1.csv")
ela222324 = pd.read_csv("./GLSearch/perf/Function_22_23_24_elaFeatures.csv")

#load performance
perf1 = pd.read_csv("./perf/all_performance.csv")


result.addPerformance(perf1)
result.addELA(ela123,ela456,ela789,ela101112,ela131415,ela161718,ela1617181,ela192021,ela1920211,ela222324 )

Xtrain, Ytest = result.createTrainSet(dataset=dataset, algorithm=None, reset=False)


model = Models(Xtrain,Ytest)
model.trainANN(inputSize=len(Xtrain[0]), dropout=0.5, hidden=0.75, epoch=100, learning=0.001, output_size=len(Ytest[0]), dataset=dataset)