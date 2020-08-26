import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import plot_model
import pandas as pd
import math
import tensorflow as tf
from keras.callbacks import CSVLogger
#import autosklearn.regression
import pickle

class Models():
    def __init__(self, features, y_cost):
        self.features = features
        self.y_cost = y_cost

    def weightedCategoricalCrossentropy(self, y_true, y_pred):
    #    y_cost = y_true * y_pred
    #	y_cost = (y_true-y_label) * 1e5
    #	condition = tf.equal(y_pred, tf.math.reduce_max(y_pred, axis=1, keepdims=True))
    #	probabilityMatrix = tf.where(condition, tf.zeros_like(y_pred), y_pred)
        
    #	costMatrix = y_cost * probabilityMatrix

    #	cost = y_label * K.log(y_pred) - costMatrix 
        return K.sum(y_true * y_pred, axis=-1,keepdims=True)


    def trainANN(self, inputSize, dropout, hidden, epoch, dataset, learning=0.001, output_size=4):
        model_name = '_Drop'+str(dropout)+'_Hidden'+str(hidden)+'_Epoch'+str(epoch)+'_Learning'+str(learning)+'_Dataset:'+dataset
        csv_logger = CSVLogger('./perf/'+model_name , separator=',', append=False)
        model = Sequential()
        model.add(Dense(int(round(inputSize*hidden/dropout,0)), input_shape=(inputSize,), activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(int(round(inputSize*hidden/dropout,0)), activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(int(round(inputSize*hidden/dropout,0)), activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(output_size, activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss=self.weightedCategoricalCrossentropy, optimizer=opt, metrics=['accuracy', self.weightedCategoricalCrossentropy])
        model.summary()

        model.fit(self.features, self.y_cost, epochs=epoch, callbacks=[csv_logger])
        model.save('./models/'+model_name)

