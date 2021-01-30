import pandas as pd
import numpy as np
from src.result import Result
import re
import os
from src.problem import Problem
from pflacco.pflacco import calculate_feature_set, create_feature_object
from src.interface import y_labels, x_labels
import keras.backend as K
import tensorflow as tf
from src.suites import Suites
from src.logger import Performance

def weightedCategoricalCrossentropy( y_true, y_pred):
        return K.mean(K.sum(y_true*y_pred, axis=1))

esconfig = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1] 

models = [
    {
        'name': 'Expected_Loss_50D_Sample',
        'model': model = tf.keras.models.load_model('./models/_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:0_Loss_WCategoricalCrossentropy', custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})
        'size': 50
    },
    {
        'name': 'Expected_Loss_100D_Sample',
        'model': model = tf.keras.models.load_model('./models/_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:1_Loss_WCategoricalCrossentropy', custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})
        'size': 100
    },    
    {
        'name': 'Expected_Loss_200D_Sample',
        'model': model = tf.keras.models.load_model('./models/_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:2_Loss_WCategoricalCrossentropy', custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})
        'size': 200
    },
    {
        'name': 'Cat_Loss_50D_Sample',
        'model': model = tf.keras.models.load_model('./models/_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:0_Loss_categorical_crossentropy'),
        'size': 50
    },
    {
        'name': 'Cat_Loss_100D_Sample',
        'model': model = tf.keras.models.load_model('./models/_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:1_Loss_categorical_crossentropy')
        'size': 100
    },
    {
        'name': 'Cat_Loss_200D_Sample',
        'model': model = tf.keras.models.load_model('./models/_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:2_Loss_categorical_crossentropy')
        'size': 200
    }
]

performanceBenchmark = Performance()
#create a benchmark for function 1 with dimensions 2 and 3. 
for i in range(1,2):
    suite = Suites(instances=[6,7,8,9,10], baseBudget=10000, dimensions=[2,3], esconfig=esconfig, function=i, performance=performanceBenchmark , pflacco=True, localSearch=None)
    suite.runTestMultipleModel(models=models, stepSize=2, precision=1e-2)
    performanceBenchmark.saveToCSVPerformance('Benchmark_Resting_with_Models_func_'+str(1))