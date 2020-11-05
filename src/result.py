import numpy as np
import pandas as pd
from src.interface import y_labels, x_labels
import math
import bbobbenchmarks.bbobbenchmarks as bn

class Result():
    def __init__(self, bestTiming= True, restricted=False):
        self.consolidatedPerformance = pd.DataFrame()
        self.processedPerformance = pd.DataFrame()
        self.calculatedPerformance = pd.DataFrame()
        self.elaFeatures = pd.DataFrame(columns=x_labels)
        self.processedFeatures = pd.DataFrame()
        self.trainingData = pd.DataFrame()
        self.classificationCost = pd.DataFrame()
        self.processedPerf = False
        self.processedSolvers = False
        self.processedELA = False
        self.bestTiming = bestTiming
        self.restricted = restricted
    
    def addPerformance(self, *args):
        dataframes = []
        for arg in args:
            dataframes.append(arg)
        
        dataframes.append(self.consolidatedPerformance)
        self.consolidatedPerformance = pd.concat(dataframes)

    def addELA(self, *args):
        dataframes = []
        for arg in args:
            dataframes.append(arg)
        
        dataframes.append(self.elaFeatures)
        self.elaFeatures = pd.concat(dataframes)


    def _elaPrecalculate(self, dataset):
        #Non destructive preprocessing
        self.processedFeatures = self.elaFeatures

        self.processedFeatures['dim'] = self.processedFeatures['name'].apply(lambda x: x.split("_")[-1])
        self.processedFeatures = self.processedFeatures[self.processedFeatures['dim']==dataset].drop(columns=['dim'])
        #self.processedFeatures['name'] = self.processedFeatures['name'].apply(lambda x: '_'.join(x.split('_')[:-3]))
        
        #split the name identifier to extract the function, instance, algorithm used and dimensions
        self.processedFeatures['function'] =  self.processedFeatures['name'].str.extract('(_F[0-9]+)')
        self.processedFeatures['function'] = self.processedFeatures['function'].apply(lambda x: x.replace('_F',''))

        self.processedFeatures['instance'] =  self.processedFeatures['name'].str.extract('(_I[0-9]+)')
        self.processedFeatures['instance'] = self.processedFeatures['instance'].apply(lambda x: x.replace('_I',''))

        self.processedFeatures['dimension'] =  self.processedFeatures['name'].str.extract('(_D[0-9]+)')
        self.processedFeatures['dimension'] = self.processedFeatures['dimension'].apply(lambda x: x.replace('_D',''))



        self.processedFeatures['trial'] =  self.processedFeatures['name'].str.extract('(_T[0-9]+)')
        self.processedFeatures['trial'] = self.processedFeatures['trial'].apply(lambda x: x.replace('_T',''))

        self.processedFeatures['budget'] = self.processedFeatures['name'].str.extract('(_B[0-9]+)')
        self.processedFeatures['budget'] = self.processedFeatures['budget'].apply(lambda x: x.replace('_B',''))


        self.processedFeatures[['function','instance', 'dimension', 'trial', 'budget']]= self.processedFeatures[['function','instance', 'dimension', 'trial', 'budget']].astype('int64')
        
        #remove duplicates because the performance table was pivoted 
        #self.processedFeatures = self.processedFeatures.groupby(['function','instance', 'dimension', 'trial', 'budget']).mean().reset_index()


        self.processedELA = True

    def _preCalculation(self):
        #Non destructive preprocessing
        self.processedPerformance = self.consolidatedPerformance
        #split the name identifier to extract the function, instance, algorithm used and dimensions
        self.processedPerformance['function'] =  self.processedPerformance['name'].str.extract('(_F[0-9]+)').astype(str)
        self.processedPerformance['function'] = self.processedPerformance['function'].apply(lambda x: x.replace('_F',''))

        self.processedPerformance['instance'] =  self.processedPerformance['name'].str.extract('(_I[0-9]+)').astype(str)
        self.processedPerformance['instance'] = self.processedPerformance['instance'].apply(lambda x: x.replace('_I',''))

        self.processedPerformance['dimension'] =  self.processedPerformance['name'].str.extract('(_D[0-9]+)').astype(str)
        self.processedPerformance['dimension'] = self.processedPerformance['dimension'].apply(lambda x: x.replace('_D',''))



        self.processedPerformance['trial'] =  self.processedPerformance['name'].str.extract('(_T[0-9]+)').astype(str)
        self.processedPerformance['trial'] = self.processedPerformance['trial'].apply(lambda x: x.replace('_T',''))

        self.processedPerformance['budget'] = self.processedPerformance['name'].str.extract('(_B[0-9]+)').astype(str)
        self.processedPerformance['budget'] = self.processedPerformance['budget'].apply(lambda x: x.replace('_B',''))

 
        self.processedPerformance['algo'] = self.processedPerformance['name'].astype(str).apply(lambda x: x[x.find('_Local')+1:x.find('_T')].replace('_',''))

        self.processedPerformance[['function','instance', 'dimension', 'trial', 'budget']]= self.processedPerformance[['function','instance', 'dimension', 'trial', 'budget']].astype('int64')

        #Calculate necessary numbers in preparation for calculate the true performance
        self.processedPerformance['ertMax'] = 10000*self.processedPerformance['dimension']
        self.processedPerformance['relERT'] = self.processedPerformance['ert-8']/self.processedPerformance['ertMax']
        maxFCE = self.processedPerformance.groupby(['function', 'instance', 'dimension','trial'])['fce'].max().reset_index()
        self.processedPerformance = self.processedPerformance.merge(maxFCE, on=['function', 'instance', 'dimension','trial'], how='left', suffixes=('', 'max'))

        #adjust FCE to factor in accuracy set when experiment was setup
        self.processedPerformance['fce'] =  self.processedPerformance['fce']+ 1e-8 
        self.processedPerformance['fceTarget'] = self.processedPerformance.apply(lambda x: self._getOptimalValue(x['function'], x['instance']), axis=1)
        self.processedPerformance['relFCE'] = self.processedPerformance.apply(lambda x: 1 + ((np.log10(float(x['fce'])/x['fceTarget']))/(np.log10(float(x['fcemax'])/x['fceTarget'])) ), axis=1)

        #Calculate performance based on Rajn's performance measure; 
        self.processedPerformance['performance'] = self.processedPerformance.apply(lambda x: x['relERT'] if x['relERT'] >0  else x['relFCE'], axis=1)

        self.processedPerf = True


    def _calculateBestSolvers(self):
        #Ensure no duplicate entries. Mean is the default aggregation.
        self.classificationCost = self.processedPerformance.pivot_table(index=['function', 'dimension','instance', 'trial', 'budget'], columns = 'algo', values='performance').reset_index().sort_values(['function', 'dimension','instance', 'trial', 'budget'], ascending=True)

        
        #get best performance for the base runner 
        baseRunner = self.classificationCost.groupby(['function', 'dimension','instance', 'trial'])['Local:Base'].min().reset_index()
        
        #replace base runner performance with its best performance
        self.classificationCost = self.classificationCost.drop(columns=['Local:Base'])
        self.classificationCost = self.classificationCost.merge(baseRunner, on=['function', 'dimension','instance', 'trial'])
        
        #rename the to performance column
        self.classificationCost = self.classificationCost.rename(columns={'Local:Base':'Local:Base_perf_ref','Local:bfgs':'Local:bfgs_perf', 'Local:nelder':'Local:nelder_perf' })

        #check if there is a restriction
        if self.restricted:
            usedOptions = ['Local:Base_perf_ref', 'Local:bfgs_perf']
        else:
            usedOptions = ['Local:Base_perf_ref', 'Local:bfgs_perf',  'Local:nelder_perf']

        #The base runner performance should be the correct choice till the optimal value for the local search is reached.
        self.classificationCost['Local:Base_perf'] = self.classificationCost['Local:Base_perf_ref']
        bestPathFeatures = np.append( usedOptions, ['function', 'dimension','instance', 'trial']).tolist()
        bestPath = self.classificationCost[bestPathFeatures].groupby(['function', 'dimension','instance', 'trial']).min().min(axis=1).reset_index()
        
        self.classificationCost = self.classificationCost.merge(bestPath, on= ['function','instance','dimension','trial'], how='left')
        self.classificationCost.rename(columns={0:'bestPath'}, inplace=True)
        
        
        self.classificationCost['best_choice'] = self.classificationCost[usedOptions].min(axis=1)
        
        if self.bestTiming:
            #this will set the score of the base runner to the best score until that point is reached.
            self.classificationCost['indicator'] =  self.classificationCost.apply(lambda x: True if x['best_choice']==x['bestPath'] else False, axis=1)

            f = None
            d = None
            i = None
            t = None
            state = None
            for index, row in self.classificationCost.iterrows():
                if row['function'] != f or row['dimension'] != d or row['instance'] != i or row['trial'] != t:
                    f = row['function']
                    d = row['dimension']
                    i = row['instance']
                    t = row['trial']
                    state = row['indicator']
                elif row['indicator']:
                    state = True
                
                self.classificationCost.loc[index,'indicator'] = state


            def getCorrectBasePerformance(x):
                if  x['indicator']:
                    return x['Local:Base_perf_ref']
                else:
                    return x['bestPath']


            self.classificationCost['Local:Base_perf'] = self.classificationCost.apply(getCorrectBasePerformance, axis=1)


        #calculate VBS
        self.classificationCost['vbs'] = self.classificationCost[usedOptions].min(axis=1)

        #calculate cost for each algorithm
        for option in usedOptions:
            self.classificationCost[option.replace('_ref','').replace('_perf','')] = self.classificationCost[option.replace('_ref','')] - self.classificationCost['vbs']


        self.processedSolvers = True


    def calculatePerformance(self):
        self._preCalculation()
        #remove the algorithm selected to have a meaningful performance aggregation
        self.processedPerformance['_algo'] =  self.processedPerformance['algo'].replace(to_replace=r'^Local:nelder[^-]', value='', regex=True)
        self.processedPerformance['_algo'] =  self.processedPerformance['_algo'].replace(to_replace=r'^Local:Base[^-]', value='', regex=True)
        self.processedPerformance['_algo'] =  self.processedPerformance['_algo'].replace(to_replace=r'^Local:bfgs[^-]', value='', regex=True)
        self.processedPerformance['_algo'] =  self.processedPerformance['_algo'].replace(to_replace=r'^ample[0-9]*[^-]', value='', regex=True)


        #first sort the function 
        self.processedPerformance = self.processedPerformance.sort_values(['function', 'dimension','instance', 'trial', 'budget'], ascending=True)
        base = self.processedPerformance[self.processedPerformance['algo']=='Local:Base'].groupby(['function', 'instance', 'dimension','trial'])['performance'].min().reset_index()
        base['algo'] = 'Base'


        #create algorithm based on the checkpoint
        local = self.processedPerformance[self.processedPerformance['algo'].isin(['Local:bfgs', 'Local:nelder'])]
        local['algo'] = local.apply(lambda x: x['algo'] + str(round(x['budget']/(x['dimension']*500),0)), axis=1)
        local = local.groupby(['function', 'instance', 'dimension','trial','algo'])['performance'].mean().reset_index()


        #SBS is the lowest mean score of the each algorithm performance
        sbsRunner = pd.concat([local,base]).groupby('algo')['performance'].mean().idxmin()
        sbsPerformance = pd.concat([local,base]).pivot_table(index=['function', 'dimension','instance', 'trial'], columns = 'algo', values='performance', aggfunc='mean').reset_index()


        #Dealing with null values in cases when the base run reached the optimal value before the checkpoint
        sbsPerformance[sbsRunner] =  sbsPerformance[sbsRunner].fillna(sbsPerformance['Base'])
 
        #just keep the SBS runner
        sbsPerformance = sbsPerformance[['function', 'dimension','instance', 'trial', sbsRunner]]
        sbsPerformance = sbsPerformance.rename(columns={sbsRunner:'performance'})
        runnerName = 'SBS_'+sbsRunner
        sbsPerformance['algo'] = runnerName

        #VBS-- Get the lowest score
        local = self.processedPerformance[self.processedPerformance['algo'].isin(['Local:bfgs', 'Local:nelder'])].groupby(['function', 'instance', 'dimension','trial','algo'])['performance'].min().reset_index()
        vbsPerformance = pd.concat([local,base]).groupby(['function', 'instance', 'dimension','trial'])['performance'].min().reset_index()
        vbsPerformance['algo'] =  'VBS'
        

        #get the individual model performance
        modelPerformance = self.processedPerformance[~self.processedPerformance['algo'].isin(['Local:Base', 'Local:nelder', 'Local:bfgs'])][['function', 'instance', 'dimension','trial','algo', 'performance']]
        
        allPerformance = pd.concat([sbsPerformance,vbsPerformance,modelPerformance]).pivot_table(index=['function', 'dimension','instance', 'trial'], columns = 'algo', values='performance').reset_index()

        return allPerformance

    def createTrainSet(self, dataset, algorithm=None, reset=False, interface=None, RNN=None):
        if reset:
            self._reset()
        if not self.processedPerf and not self.processedSolvers:
            self._preCalculation()
            self._calculateBestSolvers()
        elif not self.processedPerf:
            self._preCalculation()
        elif self.processedSolvers:
            self._calculateBestSolvers()

        if not self.processedELA:
            self._elaPrecalculate(dataset)

        if interface == None:
            inputInterface = x_labels
        else:
            inputInterface = interface
    
        self.trainingData = self.processedFeatures.merge(self.classificationCost, on= ['function','instance','dimension','trial', 'budget'])
        
        #add the percentage of budget used
        self.trainingData['budget.used'] = self.trainingData['budget'] / (10000*self.trainingData['dimension'])

        #some ela features are infinity. They are replaced by the average value
        self.trainingData.replace([np.inf, -np.inf], np.nan,  inplace=True)

        #get the function, instance and dimension with missing values
        for label in inputInterface:
            missingMask = self.trainingData[label].isna()
            missingList = self.trainingData[missingMask]['function'].astype(str) + '_' + self.trainingData[missingMask]['dimension'].astype(str) +'_'+ self.trainingData[missingMask]['instance'].astype(str)

            for missing in missingList.unique():
                attr = missing.split('_')

            #get average value and replace na
                meanMask = (self.trainingData[label].notnull()) & (self.trainingData['function']== int(attr[0])) & (self.trainingData['dimension']== int(attr[1])) & (self.trainingData['instance']== int(attr[2]))
                mean = self.trainingData[meanMask][label].mean()
                
            #check first if mean for that instance can be used, otherwise, the mean for the dimension will be used    
                if not math.isnan(mean):
                    self.trainingData[label] = self.trainingData[label].fillna(mean)
                else:
                    meanMask = (self.trainingData[label].notnull()) & (self.trainingData['function']== int(attr[0])) & (self.trainingData['dimension']== int(attr[1]))
                    mean = self.trainingData[meanMask][label].mean()
                    self.trainingData[label] = self.trainingData[label].fillna(mean)

        if (algorithm==None):
            training = self.trainingData
        else:
            training = self.trainingData[(self.trainingData['algo']==algorithm)]

        if RNN is None:
            Xtrain = training[inputInterface].values
            if not self.restricted:
                ycost = training[y_labels].values
            else:
                ycost = training[y_labels[:2]].values
        else:
            Xtrain, ycost = self._createRNNSet(RNN, training, inputInterface)
        
        return Xtrain, ycost


    def _createRNNSet(self, n_step, dataFrame, inputInterface):

        if self.restricted:
            y_labels_used = y_labels[:2]
        else:
            y_labels_used = y_labels

        #make sure that the dataframe is sorted
        dataFrame = dataFrame.sort_values(['function', 'dimension','instance', 'trial', 'budget'], ascending=True)
        x_arr = []
        y_arr = []
        
        functions = dataFrame['function'].unique()
        instance = dataFrame['instance'].unique()
        dimension = dataFrame['dimension'].unique()
        trial = dataFrame['trial'].unique()

        #we need to filter the running window based on Function, Dimension, Instance and Trial
        for f in functions:
            for d in dimension:
                for i in instance:
                    for t in trial:
                        subset =  dataFrame[(dataFrame['function']==f) & (dataFrame['dimension']==d) & (dataFrame['instance']==i) & (dataFrame['trial']==t)]
                        #make sure there's enough time steps otherwise exclude it.
                        if len(subset)-n_step >= 0:
                            #The last entry does not have an ELA calculation. As a result, it is automatically excluded when merged with the features.
                            for s in range(len(subset)-n_step+1):
                                x_arr.append(subset[inputInterface].iloc[s:s+n_step].values)
                                y_arr.append(subset[y_labels_used].iloc[s+n_step-1].values)
                            
        #convert to one hot encoded data
        ycost = np.zeros_like(np.array(y_arr))
        ycost[np.arange(len(np.array(y_arr))), np.array(y_arr).argmin(1)] = 1

        return np.array(x_arr), ycost

    def _reset(self):
        self.processedSolvers = False
        self.processedPerf = False
        self.processedELA = False

    def _getOptimalValue(self, functionID, instance, precision=1e-8):
        functionAttr = 'F' + str(functionID)
        function = getattr(bn, functionAttr)(instance)
        return function.getfopt() + precision