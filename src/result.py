import numpy as np
import pandas as pd
from src.interface import y_labels, x_labels
import math



class Result():
    def __init__(self):
        self.consolidatedPerformance = pd.DataFrame()
        self.processedPerformance = pd.DataFrame()
        self.calculatedPerformance = pd.DataFrame()
        self.elaFeatures = pd.DataFrame(columns=x_labels)
        self.processedFeatures = pd.DataFrame()
        self.trainingData = pd.DataFrame()
        self.SBS = ''
        self.classificationCost = pd.DataFrame()
        self.processedPerf = False
        self.processedSolvers = False
        self.processedELA = False
    
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
        self.processedFeatures['name'] = self.processedFeatures['name'].apply(lambda x: '_'.join(x.split('_')[:-3]))
        
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

        self.processedFeatures['algo'] = self.processedFeatures['name'].apply(lambda x: x[x.find('_Local')+1:x.find('_T')].replace('_',''))

        self.processedFeatures[['function','instance', 'dimension', 'trial', 'budget']]= self.processedFeatures[['function','instance', 'dimension', 'trial', 'budget']].astype('int64')
        
        #remove duplicates because the performance table was pivoted 
        self.processedFeatures = self.processedFeatures.groupby(['function','instance', 'dimension', 'trial', 'budget']).mean().reset_index()


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
        self.processedPerformance['relFCE'] = self.processedPerformance.apply(lambda x: 1 + ((np.log10(float(x['fce'])/1e-8))/(np.log10(float(x['fcemax'])/1e-8)) ), axis=1)

        #Calculate performance based on Rajn's performance measure; 
        self.processedPerformance['performance'] = self.processedPerformance.apply(lambda x: x['relERT'] if x['relERT'] >0  else x['relFCE'], axis=1)

        self.processedPerf = True


    def _calculateBestSolvers(self):
        #We need to pivot first and fill the missing values otherwise algorithm that finished first will be underrepresented.
        self.classificationCost = self.processedPerformance.pivot_table(index=['function', 'dimension','instance', 'trial', 'budget'], columns = 'algo', values='performance').reset_index().sort_values(['function', 'dimension','instance', 'trial', 'budget'], ascending=True)
        
        #rename the to performance column
        self.classificationCost = self.classificationCost.rename(columns={'Local:Base':'Local:Base_perf_ref','Local:bfgs0.1':'Local:bfgs0.1_perf', 'Local:bfgs0.3':'Local:bfgs0.3_perf', 'Local:nedler':'Local:nedler_perf' })

        #The base runner performance should be the correct choice till the optimal value for the local search is reached.
        self.classificationCost['Local:Base_perf'] = self.classificationCost['Local:Base_perf_ref']
        self.classificationCost['Local:Base_perf_ref'] = self.classificationCost['Local:Base_perf_ref'].bfill()
        bestPath = self.classificationCost[['Local:Base_perf_ref', 'Local:bfgs0.1_perf','Local:bfgs0.3_perf', 'Local:nedler_perf','function', 'dimension','instance', 'trial']].groupby(['function', 'dimension','instance', 'trial']).min().min(axis=1).reset_index()
        
        self.classificationCost = self.classificationCost.merge(bestPath, on= ['function','instance','dimension','trial'], how='left')
        self.classificationCost.rename(columns={0:'bestPath'}, inplace=True)
        
        #fill values based on performance of the runners
        self.classificationCost['Local:bfgs0.1_perf'] = self.classificationCost['Local:bfgs0.1_perf'].ffill()
        self.classificationCost['Local:bfgs0.3_perf'] = self.classificationCost['Local:bfgs0.3_perf'].ffill()
        self.classificationCost['Local:nedler_perf' ] = self.classificationCost['Local:nedler_perf' ].ffill()
        
        self.classificationCost['best_choice'] = self.classificationCost[['Local:Base_perf_ref', 'Local:bfgs0.1_perf','Local:bfgs0.3_perf',  'Local:nedler_perf']].min(axis=1)
        
        #this will set the score of the base runner to the best score until that point is reached. This should only be applied when the base runner has a null value performance.
        def getCorrectBasePerformance(x):
            if x['Local:Base_perf'] is not None:
                if  x['bestPath']==x['best_choice']:
                    return x['Local:Base_perf_ref']
                else:
                    return x['bestPath']
            else:
                return x['Local:Base_perf']

        self.classificationCost['Local:Base_perf'] = self.classificationCost.apply(getCorrectBasePerformance, axis=1)


        #calculate VBS
        self.classificationCost['vbs'] = self.classificationCost[['Local:Base_perf', 'Local:bfgs0.1_perf','Local:bfgs0.3_perf',  'Local:nedler_perf']].min(axis=1)

        #calculate cost for each algorithm
        self.classificationCost['Local:Base'] = self.classificationCost['Local:Base_perf'] - self.classificationCost['vbs']
        self.classificationCost['Local:bfgs0.1'] = self.classificationCost['Local:bfgs0.1_perf'] - self.classificationCost['vbs']
        self.classificationCost['Local:bfgs0.3'] = self.classificationCost['Local:bfgs0.3_perf'] - self.classificationCost['vbs']
        self.classificationCost['Local:nedler'] = self.classificationCost['Local:nedler_perf' ] - self.classificationCost['vbs']

        #calculate SBS and SBS-VBS-Gap
        indexSBS = np.array([self.classificationCost['Local:Base_perf'].mean(), self.classificationCost['Local:bfgs0.1_perf'].mean(), self.classificationCost['Local:bfgs0.3_perf'].mean(), self.classificationCost['Local:nedler_perf'].mean()]).argmin()
        sbsAlgo = y_labels[indexSBS]
        
        self.classificationCost['sbs_'+sbsAlgo]  = self.classificationCost[sbsAlgo+'_perf']
        self.classificationCost['VBS-SBS-Gap'] = self.classificationCost['sbs_'+sbsAlgo] - self.classificationCost['vbs']

        self.processedSolvers = True


    def calculatePerformance(self):
        self._preCalculation()
        self.classificationCost = self.processedPerformance.pivot_table(index=['function', 'dimension','instance', 'trial'], columns = 'algo', values='performance').reset_index().sort_values(['function', 'dimension','instance', 'trial'], ascending=True)

        #VBS should be based on the base runners
        baseRunners = ['Local:Test', 'Local:bfgs0.1-LHS1000', 'Local:bfgs0.1-LHS2000', 'Local:bfgs0.1-LHS5000', 'Local:bfgs0.3-LHS1000', 'Local:bfgs0.3-LHS2000', 'Local:bfgs0.3-LHS5000', 'Local:nedler-LHS1000', 'Local:nedler-LHS2000', 'Local:nedler-LHS5000']
        self.classificationCost['vbs'] = self.classificationCost[baseRunners].min(axis=1)

        sbs = self.classificationCost[baseRunners].mean(axis=0).idxmin(axis=0)
        self.classificationCost['sbs_'+str(sbs)]  = self.classificationCost[sbs]
        self.classificationCost['VBS-SBS-Gap'] = self.classificationCost['sbs_'+str(sbs)] - self.classificationCost['vbs']


        return self.classificationCost

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
            inputeInterface = x_labels
        else:
            inputeInterface = interface
    
        self.trainingData = self.processedFeatures.merge(self.classificationCost, on= ['function','instance','dimension','trial', 'budget'])
        
        #add the percentage of budget used
        self.trainingData['budget.used'] = self.trainingData['budget'] / (10000*self.trainingData['dimension'])
 
        #Remove first generation
        self.trainingData = self.trainingData[self.trainingData['budget'] >self.trainingData['dimension']* 500 + 400]

        #some ela features are infinity. They are replaced by the average value
        self.trainingData.replace([np.inf, -np.inf], np.nan,  inplace=True)

        #get the function, instance and dimension with missing values
        for label in inputeInterface:
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
            Xtrain = training[inputeInterface].values
            ycost = training[y_labels].values
        else:
            Xtrain, ycost = self.createRNNSet(RNN, training, inputeInterface)
        
        return Xtrain, ycost


    def createRNNSet(self, n_step, dataFrame, inputeInterface):
        x_arr = []
        y_arr = []
        functions = dataFrame['function'].unique()
        instance = dataFrame['instance'].unique()
        dimension = dataFrame['dimension'].unique()
        trial = dataFrame['trial'].unique()

        #convert to one hot encoded data
        ycost = np.zeros_like(dataFrame[y_labels].values)
        ycost[np.arange(len(dataFrame[y_labels].values)), dataFrame[y_labels].values(1)] = 1

        #we need to filter the running window based on Function, Dimension, Instance and Trial
        for f in functions:
            for d in dimension:
                for i in instance:
                    for t in trial:
                        subset =  dataFrame[(dataFrame['function']==f) & (dataFrame['dimension']==d) & (dataFrame['instance']==i) & (dataFrame['trial']==t)]
                        for i in range(len(subset)-n_step+1):
                            x_arr.append(subset[inputeInterface].iloc[i:i+n_step])
                            y_arr.append(ycost[i+n_step])
        return np.array(x_arr), np.array(y_arr)

    def _reset(self):
        self.processedSolvers = False
        self.processedPerf = False
        self.processedELA = False