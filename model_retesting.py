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


modelName = '_RNN_Hidden2_Dropout_0.2_Grossup_1_StepSize2_Epoch2000_Learning1e-05_Size:0_Loss_categorical_crossentropy'
sampleSize =50


def weightedCategoricalCrossentropy( y_true, y_pred):
        return K.mean(K.sum(y_true*y_pred, axis=1))
    
def calculateELA(sampleSize, currentResults, budget, budget_used, dimension, elaFeatures):
    sample = currentResults.iloc[:,0:dimension].values[-sampleSize:]
    obj_values = currentResults['y'].values[-sampleSize:]
    featureObj = create_feature_object(sample,obj_values, lower=-5, upper=5)

    try:
        ela_distr = calculate_feature_set(featureObj, 'ela_distr')
    except:
        ela_distr = {}


    ela_level = calculate_feature_set(featureObj, 'ela_level')


    try:
        ela_meta = calculate_feature_set(featureObj, 'ela_meta')
    except:
        ela_meta = {}

    try:
        basic = calculate_feature_set(featureObj, 'basic')
    except:
        basic ={}

    try:
        disp = calculate_feature_set(featureObj, 'disp')
    except:
        disp = {}

    try:
        limo = calculate_feature_set(featureObj, 'limo')
    except:
        limo = {}

    try:
        nbc = calculate_feature_set(featureObj, 'nbc')
    except:
        nbc = {}

    try: 
        pca = calculate_feature_set(featureObj, 'pca')
    except:
        pca ={}

    try:
        ic = calculate_feature_set(featureObj, 'ic')
    except:
        ic = {}

    ela_feat =  {**ela_distr, **ela_level, **ela_meta, **basic, **disp, **limo, **nbc, **pca, **ic }

    ela_feat['budget.used'] = budget_used / budget

    elaFeatures = elaFeatures.append(ela_feat, ignore_index=True)
    
    #to avoid any errors if the ela feature computed is infinit or null
    elaFeatures.replace([np.inf, -np.inf], np.nan,  inplace=True)
    elaFeatures = elaFeatures.fillna(0)
    
    return elaFeatures


model = tf.keras.models.load_model('./models/'+modelName, custom_objects={'weightedCategoricalCrossentropy':weightedCategoricalCrossentropy})


#load files in the benchmark data
files = os.listdir("./test")
files = sorted(files)

#just get the most latest base runner. 
files_df = pd.DataFrame()
for file in files:
    if 'Local:Base' in file:
        name = file
        function_id = int(re.search('(_F[0-9]+)', file).group(1).replace('_F',''))
        instance_id = int(re.search('(_I[0-9]+)', file).group(1).replace('_I',''))
        dim = int(re.search('(_D[0-9]+)', file).group(1).replace('_D',''))
        trial = int(re.search('(_T[0-9]+)', file).group(1).replace('_T',''))
        budget = int(re.search('(_B[0-9]+)', file).group(1).replace('_B',''))
        files_df = files_df.append({'name':name,'function': function_id, 'instance':instance_id, 'dim':dim, 'trial':trial, 'budget':budget}, ignore_index=True)


#get the latest base runner
files_df =files_df.groupby(['dim','function','instance','trial']).max().reset_index()

total = len(files_df)
for i, row_file in files_df.iterrows():
    print("In "+ row_file['name']+ '. '+str(i/total)+'% completed'.)
    #load the points
    data = pd.read_csv('./test/'+row_file['name'])
    
    #parse needed info
    function_id = int(row_file['function'])
    instance_id = int(row_file['instance'])
    dim = int(row_file['dim'])
    trial = int(row_file['trial'])
    budget = dim*10000
    budget_used = 500
    check_points= 500* dim
    check_points_iterable = range(check_points,budget,  check_points)

    elaFeatures = pd.DataFrame()
    for idx, check_point in enumerate(check_points_iterable):
        selection_data = data.iloc[:check_point]
        elaFeatures = calculateELA(sampleSize=sampleSize, currentResults=data, budget=budget, budget_used=budget_used , dimension=dim,elaFeatures=elaFeatures)

        #skip first checkpoint
        if idx > 0:
            prediction = model.predict(elaFeatures[x_labels].iloc[-2:].values.reshape(1, 2,len(x_labels)).astype('float32')).argmax()

            #if the model decides to intiate the Local search, then we get the chosen index and break the loop
            if prediction[0][0] >0:
                files_df.loc[idx,'chosen']=prediction[0][0]
                break
        #if local search has never been chosen, then we save the chosen algorithm as the CMA-ES
        elif idx ==18:
            files_df.loc[idx,'chosen']=0
