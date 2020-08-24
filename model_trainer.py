import autosklearn.regression
import pandas as pd
import numpy as np
import pickle

#choose algorithm here
algorithm= 'Local:bfgs0.1' #terminal 12
#algorithm= 'Local:bfgs0.3' #terminal 13
#algorithm= 'Local:Base' #terminal 14
#algorithm = 'Local:nedler' #terminal 15


ela50 = pd.read_csv("./train50.csv")

ela50 = ela50.drop(columns=[
    'fce_x',
    'ert',
    'ert-1',
    'ert-2',
    'ert-3',
    'ert-4',
    'ert-5',
    'ert-6',
    'ert-7',
    'ert-8',
    'opt',
    'ertMax',
    'relERT',
    'relFCE',
    'best_performance',
    'ela_meta.quad_simple.cond',
    'ic.costs_fun_evals',
    'limo.avg.length',
    'limo.avg.length.scaled',
    'limo.avg_length.norm',
    'limo.cor',
    'limo.cor.norm',
    'limo.cor.reg',
    'limo.cor.scaled',
    'limo.costs_fun_evals',
    'limo.length.sd',
    'limo.ratio.sd',
    'limo.sd.max_min_ratio',
    'limo.sd.max_min_ratio.scaled',
    'limo.sd.mean',
    'limo.sd.mean.scaled',
    'limo.sd_mean.norm',
    'limo.sd_mean.reg',
    'limo.sd_ratio.norm',
    'limo.sd_ratio.reg',
    'nbc.costs_fun_evals',
    'pca.costs_fun_evals',
    'Unnamed: 0',
    'ic.eps.ratio',
])


#remove samples with null values
columns = ela50.columns
for i in range(len(columns)):
    ela50 = ela50[ela50[columns[i]].notnull()]

#Filter the data based on the algorithm being trained
ela50 = ela50[ela50['algo']==algorithm]


#X_train = ela50.iloc[:,7:-1].values
#y_train = ela50['performance'].values

X_train = ela50.iloc[1:30000,7:-1].values
y_train = ela50['performance'].iloc[1:30000,].values

model =  autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=86400,ensemble_nbest=1,
                      ensemble_size=1, resampling_strategy='holdout')
print("training model")
model.fit(X_train, y_train)
print("Done training. Model is saved")
pickle.dump(model, open('./models/'+algorithm, 'wb'))

