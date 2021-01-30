import bbobbenchmarks.bbobbenchmarks as bn
from modea.Utils import getOpts, getVals, options,initializable_parameters, ESFitness
from modea import Algorithms, Parameters
import numpy as np
from numpy import log
from scipy.optimize import minimize, Bounds
import pandas as pd
import src.config as config
import os
from pflacco.pflacco import calculate_feature_set, create_feature_object
from pyDOE import lhs
from src.interface import y_labels, x_labels
import math

class Problem():
    def __init__(self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco, localSearch=None, precision=1e-8):
        self.pflacco = pflacco
        self.totalBudget = budget
        self.remainingBudget = budget
        self.spentBudget = 0
        self.function = function
        self.instance = instance
        self.dimension = dimension
        self.esconfig = esconfig 
        self.performance = logger
        self.checkPoint = checkPoint
        self.activeColumns = ['x'+str(x) for x in range(1,dimension+1)] + ['y', 'name']
        self.currentResults =  pd.DataFrame(columns=self.activeColumns)
        self.elaFetures =  pd.DataFrame(columns=x_labels)
        self.prevRemainingBudget = None
        self.prevSpentBudget = None
        self.localSearch = localSearch
        self.x_labels = x_labels
        self.precision=precision
        self.baseDIR = os.getcwd()

        self.ela_feat = None

        self.problemInstance = None
        self.optimizer = None
        self.optimalValue = None

        self.createProblemInstance()
        self.initializedESAlgorithm()

    def reset(self):
        self.remainingBudget = self.totalBudget
        self.spentBudget = 0
        self.currentResults =  pd.DataFrame(columns=self.activeColumns)
        self.elaFetures =  pd.DataFrame(columns=x_labels)
        self.prevRemainingBudget = None
        self.prevSpentBudget = None
        self.ela_feat = None
        self.optimizer = None
        self.initializedESAlgorithm()

    def getProblemName(self, functionID, instance, budget, local, testRun):
        functionNamesNoiseless = [ 
                '1_Noise-free_Sphere_function',
                '2_Separable_ellipsoid',
                '3_Rastrigin',
                '4_skew_Rastrigin-Bueche',
                '5_Linear_slope',
                '6_Attractive_sector_function',
                '7_Step-ellipsoid',
                '8_Rosenbrock_noise-free',
                '9_Rosenbrock_rotated',
                '10_Ellipsoid_with_monotone_transformation',
                '11_Discus_with_monotone_transformation',
                '12_Bent_cigar',
                '13_Sharp_ridge',
                '14_Sum_of_different_powers',
                '15_Rastrigin_with_asymmetric_non-linear_distortion',
                '16_Weierstrass_condition_100',
                '17_Schaffers_F7_condition_10',
                '18_Schaffers_F7_condition_1000',
                '19_sum_of_Griewank-Rosenbrock',
                '20_Schwefel_with_tridiagonal_transformation',
                '21_Gallagher_with_101_Gaussian_peaks',
                '22_Gallagher_with_21_Gaussian_peaks',
                '23_Katsuura_function',
                '24_Lunacek_bi-Rastrigin_condition_100',
                '25_The_number_of_legs_the_animal'
                ]
    
        
        esConfig = ''.join([str(elem) for elem in self.esconfig if elem is not None]) 
        if functionID >0 and functionID <25:
            functionIndex = functionID - 1
            functionAttr = functionNamesNoiseless[functionIndex]
            
            functionName = '_F'+functionAttr + '_I' + str(self.instance) + '_D' + str(self.dimension)+ '_ES'+esConfig + "_Local:" + str(local) + '_T' + str(testRun) + "_B" + str(budget)
            return functionName
        elif functionID == 0:
            functionAttr = 'Parabola'
            functionName = '_F'+str(functionID)+functionAttr + '_I' + str(self.instance) + '_D' + str(self.dimension)+ '_ES'+esConfig  + "_Local:" + str(local)  + '_T' + str(testRun) + "_B" + str(budget)

            return functionName

    def createProblemInstance(self):
        if self.function >0 and self.function <25:
            functionAttr = 'F' + str(self.function)
            function = getattr(bn, functionAttr)(self.instance)
            def functionInstance(x):
                self.remainingBudget = self.remainingBudget - 1
                self.spentBudget = self.spentBudget + 1

                result = function(x)
    
                data = {}
                data['y'] = result
                for i in range(len(x)):
                    data['x'+str(i+1)] = x[i]
                self.currentResults = self.currentResults.append(data, ignore_index=True,)
                return result
            self.problemInstance = functionInstance
            self.optimalValue = function.getfopt() + self.precision
        elif self.function == 0:
            def parabola(x):
                self.remainingBudget = self.remainingBudget - 1
                self.spentBudget = self.spentBudget + 1
                result = sum([number**2 for number in x])
                data = {}
                data['y'] = result
                for i in range(len(x)):
                    data['x'+str(i+1)] = x[i]
                self.currentResults = self.currentResults.append(data, ignore_index=True,)
                return result
            self.problemInstance = parabola 
            self.optimalValue = 0 + self.precision
    

    def runDataGathering(self, size):
        #Runs five independent tests
        for i in range(1,6):
            self.reset()
            self.runOptimizer(i, size)
    
    def runOptimizer(self, testRun, size):
        checkpoints = self.getCheckPoints()
        currentLength = 0
        maxIndex = len(checkpoints)
        targetReachedEA = False

        #Stop the iteration if target is reached OR budget is reached
        while self.totalBudget > self.spentBudget and not targetReachedEA:
            
            self._printProgressBar(self.spentBudget, self.totalBudget,prefix='Problem with '+str(self.dimension) + 'd - f'+ str(self.function) + ' - i' + str(self.instance) + ' -t' + str(testRun),length=50)

            if (checkpoints[currentLength] < self.spentBudget and currentLength < maxIndex):
                currentLength += 1

                name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
                self.calculatePerformance(name)
                # Get the best individuals as of this time as input to the local search. Calculate the ELA features
                x0 = np.array(self.optimizer.best_individual.genotype.flatten())

                if self.pflacco:
                    for i, s in enumerate(size):
                        elaName = self.getProblemName(self.function, self.instance, self.spentBudget,'NONE',testRun)+ '_ela_' + str(i)
                        self.calculateELA(size=s)
                        self.saveElaFeat(elaName)

                self.saveState()
                
                #Simplex Method
                name = self.getProblemName(self.function, self.instance, self.spentBudget,'nelder',testRun)
                
                self.simplexAlgorithm(x0)

                self.calculatePerformance(name)
                self.loadState()
                
                name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs',testRun)

                self.bfgsAlgorithm(x0)

                self.calculatePerformance(name)

                self.loadState()

                
            
            if round(self.optimizer.best_individual.fitness,8)<=self.optimalValue and not targetReachedEA:
                targetReachedEA = True
                name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
        
                #self.currentResults['name'] = name
                #self.currentResults.to_csv('temp/'+name+'.csv',index=False)
                #self.performance.importHistoricalPath('temp/'+name+'.csv')
                #self.performance.saveToCSVPerformance('Function_'+str(self.function))
            
            #If the optimal value is not reached then continue running
            self.optimizer.runOneGeneration()
            self.optimizer.recordStatistics()


        
        name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
        self.calculatePerformance(name)


    def runTest(self):
        #Runs five independent tests
        for i in range(1,6):
            self.reset()
            self.runPerformanceTest(i)
            name = self.getProblemName(self.function, self.instance, self.spentBudget, 'test', str(i))
            self.checkDirectory(self.baseDIR+'/currentPoints')
            self.currentResults.to_csv(self.baseDIR+'/currentPoints/'+name+'.csv',index=False)

    def runPerformanceTest(self, testRun):
        currentLength = 0
        targetReachedEA = False
        #Stop the iteration if target is reached OR budget is reached
        if (self.localSearch ==  None):
            while self.totalBudget > self.spentBudget and not (targetReachedEA):
                self._printProgressBar(self.spentBudget, self.totalBudget,prefix='Problem with '+str(self.dimension) + 'd - f'+ str(self.function) + ' - i' + str(self.instance) + ' -t' + str(testRun),length=50)
                currentLength += 1
                if round(self.optimizer.best_individual.fitness,8)<=self.optimalValue and not targetReachedEA:
                    targetReachedEA = True
                    
                    #If the optimal value is not reached then continue running
                self.optimizer.runOneGeneration()
                self.optimizer.recordStatistics()
            name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Test',testRun)
            self.currentResults['name'] = name
            self.checkDirectory(self.baseDIR+'/temp')
            self.currentResults.to_csv(self.baseDIR+'/temp/'+name+'.csv',index=False)
            self.performance.importHistoricalPath('temp/'+name+'.csv')
            
        elif self.localSearch=='bfgs':
            for i in [1000,2000,5000]:
                print('Running test using local search bfgs on function '+str(self.function) +' with instance '+str(self.instance) + ' dimension '+str(self.dimension) + ' LHS Run '+str(i))
                x0 = self.generateLHSBestIndividuals(i)
                self.bfgsAlgorithm(x0)
                name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs_LHS'+str(i)+'_',testRun)
                _ = self.calculatePerformance(name)
                self.currentResults['name'] = name
                self.checkDirectory(self.baseDIR+'/temp')
                self.currentResults.to_csv(self.baseDIR+'/temp/'+name+'.csv',index=False)
                self.performance.importHistoricalPath('temp/'+name+'.csv')

        
        elif self.localSearch=='nelder':
            for i in [1000,2000,5000]:
                print('Running test using local search nelder on function '+str(self.function) +' with instance '+str(self.instance) + ' dimension '+str(self.dimension)+ ' LHS Run '+str(i))
                x0 = self.generateLHSBestIndividuals(i)
                self.simplexAlgorithm(x0)
                name = self.getProblemName(self.function, self.instance, self.spentBudget,'nelder_LHS'+str(i)+'_',testRun)
                _ = self.calculatePerformance(name)
                self.currentResults['name'] = name
                self.checkDirectory(self.baseDIR+'/temp')
                self.currentResults.to_csv(self.baseDIR+'/temp/'+name+'.csv',index=False)
                self.performance.importHistoricalPath('temp/'+name+'.csv')

    def saveState(self):
        temp = 'F_' + str(self.function) +'_I_'+ str(self.instance) +'_D_'+ str(self.dimension)+'.csv'
        self.prevRemainingBudget  = self.remainingBudget 
        self.prevSpentBudget  = self.spentBudget 
        self.checkDirectory(self.baseDIR+'/temp1')
        self.currentResults.to_csv(self.baseDIR+'/temp1/'+temp, index=False)

    def loadState(self):
        temp = 'F_' + str(self.function) +'_I_'+ str(self.instance) +'_D_'+ str(self.dimension)+'.csv'
        self.remainingBudget = self.prevRemainingBudget
        self.spentBudget = self.prevSpentBudget 
        self.currentResults = pd.read_csv('temp1/'+ temp)

    def initializedESAlgorithm(self):
        representation = self.ensureFullLengthRepresentation(self.esconfig)
        opts = getOpts(representation[:len(options)])
        values = getVals(representation[len(options)+2:])
        values = getVals(self.esconfig)

        customES = Algorithms.CustomizedES(self.dimension, self.problemInstance, budget=self.totalBudget, opts=opts, values=values)

        customES.mutateParameters = customES.parameters.adaptCovarianceMatrix

        self.optimizer = customES


    def ensureFullLengthRepresentation(self, representation):
        """
            Given a (partial) representation, ensure that it is padded to become a full length customizedES representation,
            consisting of the required number of structure, population and parameter values.

            >>> ensureFullLengthRepresentation([])
            [0,0,0,0,0,0,0,0,0,0,0, None,None, None,None,None,None,None,None,None,None,None,None,None,None,None]

            :param representation:  List representation of a customizedES instance to check and pad if needed
            :return:                Guaranteed full-length version of the representation

            This function comes from https://github.com/sjvrijn/ConfiguringCMAES unmodified.
        """
        default_rep = [0]*len(options) + [None, None] + [None]*len(initializable_parameters)
        if len(representation) < len(default_rep):
            representation.extend(default_rep[len(representation):])
        return representation

    def getCheckPoints(self):
        checkpoints = range(self.checkPoint, self.totalBudget + self.checkPoint , self.checkPoint)
        return checkpoints

    def simplexAlgorithm(self, population):
        maxiter = self.remainingBudget
        #x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
        opt={'maxfev': maxiter, 'disp': False, 'return_all': False}
        minimize(self.problemInstance,tol=self.precision, x0=population, method='nelder-mead', options=opt)

    def bfgsAlgorithm(self, population):
        #x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
        opt={'maxiter' : self.remainingBudget, 'disp': False, 'return_all': False}
        
        minimize(self.problemInstance,tol=self.precision,  x0=population, method='BFGS', options=opt)
    
    def calculateELA(self, size=None, sanitize=False):
        if size is None:
            sample = self.currentResults.iloc[:,0:self.dimension].values
            obj_values = self.currentResults['y'].values
            featureObj = create_feature_object(sample,obj_values, lower=-5, upper=5)
        else:
            #remove values that are out of the lower and upper bound that are sometimes generated by MODEA
            for column in self.activeColumns[0:-2]:
                self.currentResults = self.currentResults[(self.currentResults[column]>-5) & (self.currentResults[column]<5)]
            
            if size == '5G':
                genSize = 4+ math.floor(3*log(self.dimension))
                sampleSize =  genSize * 5 * self.dimension
            elif size == '50G':
                genSize = 4+ math.floor(3*log(self.dimension))
                sampleSize =  genSize * 50 * self.dimension
            elif size == '100G':
                genSize = 4+ math.floor(3*log(self.dimension))
                sampleSize =  genSize * 100 * self.dimension
            else:
                sampleSize = self.dimension * size 
            
            sample = self.currentResults.iloc[:,0:self.dimension].values[-sampleSize:]
            obj_values = self.currentResults['y'].values[-sampleSize:]
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

        self.ela_feat =  {**ela_distr, **ela_level, **ela_meta, **basic, **disp, **limo, **nbc, **pca, **ic }

        self.ela_feat['budget.used'] = self.spentBudget / self.totalBudget

        self.elaFetures = self.elaFetures.append(self.ela_feat, ignore_index=True)
        
        if sanitize == True:
            self.sanitizeELAFeatures()

    def saveElaFeat(self, name):
        #If Pflacco is bifurcated, it will save a csv of the current results before the local search
        if self.pflacco:
            self.performance.insertELAData(name, self.ela_feat)
        else:
            self.checkDirectory(self.baseDIR+'/temp')
            self.currentResults.to_csv(self.baseDIR+'temp/'+name+'_pflacco.csv',index=False)
    
    def calculatePerformance(self, name, save=True):
        optimalValue = self.optimalValue  
        ert8, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue)-(self.precision)+(1e-8))

        ert7, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-7)))

        ert6, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-6)))

        ert5, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-5)))

        ert4, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-4)))

        ert3, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-3)))

        ert2, fce, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-2)))

        ert1, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e-1)))
        
        ert0, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e0)))

        ertp1, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e1)))

        ertp2, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e2)))

        ertp3, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e3)))

        ertp4, _, _, _, _ = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(self.precision)+(1e4)))
        
        if save:
            self.performance.insertPerformance(name= name, ert8=ert8, ert7=ert7, ert6=ert6, ert5=ert5, ert4=ert4, ert3=ert3, ert2=ert2, ert1=ert1, ert0=ert0, ertp1=ertp1, ertp2=ertp2, ertp3=ertp3, ertp4=ertp4, fce=fce)

        return minValue
    
    def _printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
            Source: https://stackoverflow.com/questions/62116732/combining-two-for-to-integrate-a-progress-bar-while-deleting-line-in-a-text-fi
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()


    def _calcFCEandERT(self, fitnesses, target):
        """
            Calculates the FCE and ERT of a given set of function evaluation results and target value

            :param fitnesses:   Numpy array of size (num_runs, num_evals)
            :param target:      Target value to use for basing the ERT on. 
            :return:            ESFitness object with FCE and ERT properly set
            Source: Modea Python Package by https://github.com/sjvrijn, slightly modified to consider target based on BBOB target value.
        """
        min_fitnesses = np.min(fitnesses, axis=1).tolist()  # Save as list to ensure eval() can read it as summary

        num_runs, num_evals = fitnesses.shape
        below_target = fitnesses <= target
        num_below_target = np.sum(below_target, axis=1)
        min_indices = []
        num_successful = 0
        for i in range(num_runs):
            if num_below_target[i] != 0:
                # Take the lowest index at which the target was reached.
                min_index = np.min(np.argwhere(below_target[i]))
                num_successful += 1
            else:
                # No evaluation reached the target in this run
                min_index = num_evals
            min_indices.append(min_index)

        min_fixed_error =[round(x-target,8) for x in min_fitnesses]
        

        minValue = round(np.min(fitnesses),8)
        FCE = np.mean(min_fixed_error) 
        std_dev_FCE = np.std(min_fixed_error)

        ### ERT ###
        # If none of the runs reached the target, there is no (useful) ERT to be calculated
        ERT = np.sum(min_indices) / num_successful if num_successful != 0 else None
        std_dev_ERT = np.std(min_indices)

        return ERT, FCE, std_dev_ERT, std_dev_FCE, minValue

    def generateLHSBestIndividuals(self, samples):
        sample = lhs(self.dimension, samples=samples)*10-5
        bestValue = np.inf
        bestgenoType = None

        for i in sample:
            currentRes = self.problemInstance(i)
            if currentRes < bestValue:
                bestValue = currentRes
                bestgenoType = i
        return bestgenoType

    def runASPBattery(self,ASP, ASPName, size,  stepSize, restart= False, features=None):
        #Runs five independent tests
        for i in range(1,6):
            self.reset()
            self.runASPTest(i, ASP,ASPName, size, restart, features, stepSize)
            name = self.getProblemName(self.function, self.instance, self.spentBudget, ASPName, str(i))
            self.checkDirectory(self.baseDIR+'/currentPoints')
            self.currentResults.to_csv(self.baseDIR+'/currentPoints/'+name+'.csv',index=False)


    def runASPTest(self, testRun, ASP,ASPName, size, restart, features, stepSize):
        checkpoints = self.getCheckPoints()
        currentLength = 0
        maxIndex = len(checkpoints)-1
        targetReached = False
        if features is not None:
            x_labels = features
        else:
            x_labels = self.x_labels 

        #default name is the base. If the algorithm selects the local search, the name will be overridden.
        name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base'+ASPName,testRun)
                

        #Run model ES algorithm
        while self.totalBudget > self.spentBudget and not (targetReached):
            self._printProgressBar(self.spentBudget, self.totalBudget,prefix='Problem with '+str(self.dimension) + 'd - f'+ str(self.function) + ' - i' + str(self.instance) + ' -t' + str(testRun),length=50)

            #If the optimal value is not reached then continue running
            self.optimizer.runOneGeneration()
            self.optimizer.recordStatistics()
            
            #Check the check point then calculate the ELA
            if (checkpoints[currentLength] < self.spentBudget and currentLength < maxIndex):
                currentLength += 1
                self.calculateELA(size=size, sanitize=True)
                print(len(self.elaFetures))
                
                if "RNN" in ASPName:
                    print("calulating ELA ASP")
                    #We need to have at least the number of step size
                    if len(self.elaFetures) < stepSize:
                        index = 0
                    else:  
                        ela = np.array([self.elaFetures[x_labels].iloc[-1,].values]).astype('float32')
        
                        #add additional step in the ela
                        for i in range(2, stepSize+1):   
                            ela1 = np.array([self.elaFetures[x_labels].iloc[-i,].values]).astype('float32')
                            ela = np.concatenate((ela1,ela),axis=0)
        
                        print(ela)
                        index = ASP.predict(ela.reshape(1, stepSize,len(x_labels)).astype('float32')).argmax()

                else:
                    
                    ela = self.elaFetures[x_labels].iloc[-1,]
                    print(ela)
                    index = ASP.predict(ela.values.reshape(1,-1)).argmax()
                
                print("Selected algorihtm "+ y_labels[index])


                #if index is greater than 0, then local search must be used
                if (index > 0):
                    x0 = np.array(self.optimizer.best_individual.genotype.flatten())

                     #Check if BFGS
                    if (index==1):
                        name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs'+ASPName,testRun)
                        #self.saveElaFeat(name)
                        self.bfgsAlgorithm(x0)

                    if (index == 2):
                        name = self.getProblemName(self.function, self.instance, self.spentBudget,'nelder'+ASPName,testRun)
                        self.simplexAlgorithm(x0)


                    if not restart:
                        targetReached = True
                
                #check if the target is reached
                if round(self.optimizer.best_individual.fitness,8)<=self.optimalValue:
                    targetReached = True

        _ = self.calculatePerformance(name)
        
        #self.currentResults['name'] = name
        #self.currentResults.to_csv('test/'+name+'.csv',index=False)
        #self.performance.importHistoricalPath('test/'+name+'.csv')
        #self.performance.saveToCSVPerformance('Test_'+name)

    def sanitizeELAFeatures(self):
        #some ela features are infinity. They are replaced by the average value
        self.elaFetures.replace([np.inf, -np.inf], np.nan,  inplace=True)

        #get the function, instance and dimension with missing values
        for label in self.x_labels:
            missingMask = self.elaFetures[label].isna()
            missingList = self.elaFetures[missingMask].values

            #check if there are missing values otherwise, no cleaning needed
            if len(missingList)>0:
                #get average value and replace na
                mean = self.elaFetures[label].mean()
                self.elaFetures[label] = self.elaFetures[label].fillna(mean)
                
            #check if the value is still missing. Impute a value of 0 if there is still nan.
            if (np.isnan(sum(self.elaFetures[label]))):
                self.elaFetures[label] = self.elaFetures[label].fillna(0)

    def checkDirectory(self, path):
        if not os.path.exists(str(path)):
            os.mkdir(path)


    def runTestMultipleModels(self,models, stepSize, features=None):
        #models must be a list of dictionary of name and model.
        #Runs five independent tests
        for i in range(1,6):
            self.reset()
            self.runMultipleModels(i, models, features, stepSize)
            name = self.getProblemName(self.function, self.instance, self.spentBudget, 'models', str(i))
            self.checkDirectory(self.baseDIR+'/currentPoints')
            self.currentResults.to_csv(self.baseDIR+'/currentPoints/'+name+'.csv',index=False)


    def runMultipleModels(self, testRun, models, features, stepSize):
        selected_checkpoint = pd.DataFrame()
        checkpoints = self.getCheckPoints()
        currentLength = 0
        maxIndex = len(checkpoints)-1
        targetReached = False
        if features is not None:
            x_labels = features
        else:
            x_labels = self.x_labels 

        #default name is the base. If the algorithm selects the local search, the name will be overridden.
        #name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base'+ASPName,testRun)
                

        #Run model ES algorithm
        while self.totalBudget > self.spentBudget and not (targetReached):
            self._printProgressBar(self.spentBudget, self.totalBudget,prefix='Problem with '+str(self.dimension) + 'd - f'+ str(self.function) + ' - i' + str(self.instance) + ' -t' + str(testRun),length=50)

            #If the optimal value is not reached then continue running
            self.optimizer.runOneGeneration()
            self.optimizer.recordStatistics()
            
            #Check the check point then calculate the ELA
            if (checkpoints[currentLength] < self.spentBudget and currentLength < maxIndex):
                currentLength += 1

                sizes = [50,100,200]
                for size in sizes:
                    self.calculateELA(size=size, sanitize=True)
                    print(len(self.elaFetures))
                    
                    #ELA features will only be computed if there are models in the model list
                    for model in models:
                        if model['size']==size:
                            if "RNN" in model['name']:
                                print("calulating ELA ASP")
                                #We need to have at least the number of step size
                                if len(self.elaFetures) < stepSize:
                                    index = 0
                                else:  
                                    ela = np.array([self.elaFetures[x_labels].iloc[-1,].values]).astype('float32')
                    
                                    #add additional step in the ela
                                    for i in range(2, stepSize+1):   
                                        ela1 = np.array([self.elaFetures[x_labels].iloc[-i,].values]).astype('float32')
                                        ela = np.concatenate((ela1,ela),axis=0)
                    
                                    print(ela)
                                    index = model['asp'].predict(ela.reshape(1, stepSize,len(x_labels)).astype('float32')).argmax()
                            else:
                                #just continue
                                index =0
                            print("Selected algorihtm of "+ model['name'] +' is '+ y_labels[index])

                            #if index is greater than 0, then local search must be used
                            if (index > 0):
                                if (index==1):
                                    selected_checkpoint = selected_checkpoint.append({'model':model['name'], 'function':self.function, 'instance': self.instance, 'budget': self.spentBudget,'local':'BFGS' }, ignore_index=True)

                                if (index == 2):
                                    selected_checkpoint = selected_checkpoint.append({'model':model['name'], 'function':self.function, 'instance': self.instance, 'budget': self.spentBudget,'local':'Nelder' }, ignore_index=True)

                                #remove the model from the models
                                models.remove(model)


                
                #check if the target is reached
                if round(self.optimizer.best_individual.fitness,8)<=self.optimalValue:
                    targetReached = True


                name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
                self.calculatePerformance(name, save=False)
                # Get the best individuals as of this time as input to the local search. Calculate the ELA features
                x0 = np.array(self.optimizer.best_individual.genotype.flatten())

                self.saveState()
                
                #Simplex Method
                name = self.getProblemName(self.function, self.instance, self.spentBudget,'nelder',testRun)
                
                self.simplexAlgorithm(x0)

                self.calculatePerformance(name)
                self.loadState()
                
                name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs',testRun)

                self.bfgsAlgorithm(x0)

                self.calculatePerformance(name)

                self.loadState()

                
            
            if round(self.optimizer.best_individual.fitness,8)<=self.optimalValue and not targetReached:
                targetReached = True
                name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
            
                    #self.currentResults['name'] = name
                    #self.currentResults.to_csv('temp/'+name+'.csv',index=False)
                    #self.performance.importHistoricalPath('temp/'+name+'.csv')
                    #self.performance.saveToCSVPerformance('Function_'+str(self.function))
                
                #If the optimal value is not reached then continue running
                self.optimizer.runOneGeneration()
                self.optimizer.recordStatistics()


        
        name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
        self.calculatePerformance(name)