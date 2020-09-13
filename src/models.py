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
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.feature_selection import RFE
from src.interface import y_labels, x_labels
from sklearn.utils import shuffle

class Models():
    def __init__(self, features, y_cost):
        #shuffle the data upon loading
        self.features, self.y_cost = shuffle(features, y_cost)
        self.y_class = None

    def weightedCategoricalCrossentropy(self, y_true, y_pred):
    #    y_cost = y_true * y_pred
    #	y_cost = (y_true-y_label) * 1e5
    #	condition = tf.equal(y_pred, tf.math.reduce_max(y_pred, axis=1, keepdims=True))
    #	probabilityMatrix = tf.where(condition, tf.zeros_like(y_pred), y_pred)

    #	costMatrix = y_cost * probabilityMatrix

    #	cost = y_label * K.log(y_pred) - costMatrix 
        return K.sum(K.sum(y_true*y_pred, axis=1))

    def trainANN(self, inputSize, dropout, hidden, epoch, size, learning=0.001, output_size=4, loss='WCategoricalCrossentropy'):
        if loss == 'WCategoricalCrossentropy':
            lossFunc = self.weightedCategoricalCrossentropy
            y_true = self.y_cost
        else:
            lossFunc = loss
            self.oneHotEncode()
            y_true = self.y_class
        model_name = '_Drop'+str(dropout)+'_Hidden'+str(hidden)+'_Epoch'+str(epoch)+'_Learning'+str(learning)+'_Size:'+str(size)+'_Loss'+loss
        csv_logger = CSVLogger('./perf/'+model_name , separator=',', append=False)
        model = Sequential()
        model.add(Dense(int(round(inputSize*hidden/dropout,0)), input_shape=(inputSize,), activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(int(round(inputSize*hidden,0)), activation='relu'))
        model.add(Dense(int(round(inputSize*hidden,0)), activation='relu'))
        model.add(Dense(output_size, activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)

        model.compile(loss=lossFunc, optimizer=opt)
        model.summary()

        model.fit(self.features, y_true, epochs=epoch, callbacks=[csv_logger])
        model.save('./models/'+model_name)

    def inferClass(self):
        #convert cost to class-- the algorithm with the least cost is the optimal cost
        self.y_class = self.y_cost.argmin(1)

    def oneHotEncode(self):
        self.y_class = np.zeros_like(self.y_cost)
        self.y_class[np.arange(len(self.y_cost)), self.y_cost.argmin(1)] = 1

    def trainRandomForest(self, size, selection=True):
        self.inferClass()
        model = RandomForestClassifier(n_estimators=500)
        if selection:
            print("Features are being selectioned")
            name = "randomForest_Selection_" + size
            selector = RFE(model, n_features_to_select=15, step=1)
            selector = selector.fit(self.features, self.y_class)
            selectedFeaturesIndex = selector.support_
            selectedFeatures =  np.array(x_labels)[selectedFeaturesIndex]
            features = np.array([])
            for arr in self.features:
                features = np.append(features, arr[selectedFeaturesIndex])
            features = features.reshape(-1,15)

        else:
            name = "randomForest_noSelection" + size
            numFeatures = len(self.features)
            features = self.features
            selectedFeatures = x_labels

        print("Training the model")
        model.fit(features, self.y_class)
        pickle.dump(model, open('./models/'+name, 'wb'))
        np.save('./models/'+name+'feat', selectedFeatures)