import bbobbenchmarks.bbobbenchmarks as bn
from modea.Utils import getOpts, getVals, options,initializable_parameters, ESFitness
from modea import Algorithms, Parameters
import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
import src.config as config
import os
from pflacco.pflacco import calculate_feature_set, create_feature_object

class Problem():
	def __init__(self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco):
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
		self.elaFetures =  pd.DataFrame()
		self.prevRemainingBudget = None
		self.prevSpentBudget = None

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
		self.elaFetures =  pd.DataFrame()
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
			self.optimalValue = function.getfopt() + 1e-8
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
			self.optimalValue = 0 + 1e-8
	

	def runTest(self):
		#Runs five independent tests
		for i in range(1,6):
			self.reset()
			self.runOptimizer(i)
	
	def runOptimizer(self, testRun):
		checkpoints = self.getCheckPoints()
		currentLength = 0
		maxIndex = len(checkpoints)
		targetReachedEA = False
		targetReachedSimplex = False
		targetReachedBFGS10 = False
		targetReachedBFGS30 = False

		#Stop the iteration if target is reached OR budget is reached
		while self.totalBudget > self.spentBudget and not (targetReachedEA and targetReachedSimplex and targetReachedBFGS10 and targetReachedBFGS30):
			
			self._printProgressBar(self.spentBudget, self.totalBudget,prefix='Problem with '+str(self.dimension) + 'd - f'+ str(self.function) + ' - i' + str(self.instance) + ' -t' + str(testRun),length=50)

			if (checkpoints[currentLength] < self.spentBudget and currentLength < maxIndex and not (targetReachedSimplex and targetReachedBFGS10 and targetReachedBFGS30)):
				currentLength += 1
				# Get the best individuals as of this time as input to the local search. Calculate the ELA features
				x0 = np.array(self.optimizer.best_individual.genotype.flatten())

				if self.pflacco:
					self.calculateELA()


				self.saveState()
				
				#Simplex Method
				if (not targetReachedSimplex):
					name = self.getProblemName(self.function, self.instance, self.spentBudget,'nedler',testRun)
					
					self.saveElaFeat(name)
					
					self.simplexAlgorithm(x0)

					minPerformance = self.calculatePerformance(name)
					self.currentResults['name'] = name
					self.currentResults.to_csv('temp/'+name+'.csv',index=False)
					self.performance.importHistoricalPath('temp/'+name+'.csv')
					self.performance.saveToCSV('Function_'+str(self.function))

					# If target is reached, we stop the calculation to save on CPU power
					if minPerformance <= self.optimalValue:
						targetReachedSimplex = True
					
					self.loadState()
				

				#Gradient Descent Method 0.1
				if (not targetReachedBFGS10):
					name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs0.1',testRun)
					self.saveElaFeat(name)
					self.bfgsAlgorithm(x0, 0.1)

					minPerformance = self.calculatePerformance(name)
					self.currentResults['name'] = name
					self.currentResults.to_csv('temp/'+name+'.csv',index=False)
					self.performance.importHistoricalPath('temp/'+name+'.csv')
					self.performance.saveToCSV('Function_'+str(self.function))

					# If target is reached, we stop the calculation to save on CPU power
					if minPerformance <= self.optimalValue:
						targetReachedBFGS10 = True
					
					self.loadState()

				#Gradient Descent Method 0.3
				if (not targetReachedBFGS30):
					name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs0.3',testRun)
					self.saveElaFeat(name)
					self.bfgsAlgorithm(x0, 0.3)

					self.calculatePerformance(name)
					self.currentResults['name'] = name
					self.currentResults.to_csv('temp/'+name+'.csv',index=False)
					self.performance.importHistoricalPath('temp/'+name+'.csv')
					self.performance.saveToCSV('Function_'+str(self.function))

					# If target is reached, we stop the calculation to save on CPU power
					if minPerformance <= self.optimalValue:
						targetReachedBFGS30 = True
					
					self.loadState()
				
			
			if round(self.optimizer.best_individual.fitness,8)<=self.optimalValue and not targetReachedEA:
				targetReachedEA = True
				name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)
		
				self.currentResults['name'] = name
				minPerformance = self.calculatePerformance(name)
				self.currentResults.to_csv('temp/'+name+'.csv',index=False)
				self.performance.importHistoricalPath('temp/'+name+'.csv')
				self.performance.saveToCSV('Function_'+str(self.function))
			
			#If the optimal value is not reached then continue running
			self.optimizer.runOneGeneration()
			self.optimizer.recordStatistics()


		
		name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base',testRun)

		self.currentResults['name'] = name
		minPerformance = self.calculatePerformance(name)
		self.currentResults.to_csv('temp/'+name+'.csv',index=False)
		self.performance.importHistoricalPath('temp/'+name+'.csv')
		self.performance.saveToCSV('Function_'+str(self.function))



	def saveState(self):
		temp = 'F_' + str(self.function) +'_I_'+ str(self.instance) +'_D_'+ str(self.dimension)+'.csv'
		self.prevRemainingBudget  = self.remainingBudget 
		self.prevSpentBudget  = self.spentBudget 
		self.currentResults.to_csv('temp1/'+temp, index=False)

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
		checkpoints = range(1, self.totalBudget + self.checkPoint , self.checkPoint)
		return checkpoints

	def simplexAlgorithm(self, population):
		maxiter = self.remainingBudget
		#x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
		opt={'maxfev': maxiter, 'disp': False, 'return_all': False}

		#minimize(self.problemInstance, x0=population, method='nelder-mead', bounds = x_bounds, options=opt)
		minimize(self.problemInstance, x0=population, method='nelder-mead', options=opt)

	def bfgsAlgorithm(self, population, stepSize):
		#x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
		opt={'maxiter' : self.remainingBudget, 'disp': False, 'return_all': False, 'eps': stepSize}

		#minimize(self.problemInstance,tol=1e-8,  x0=population, method='BFGS', bounds = x_bounds, options=opt)
		minimize(self.problemInstance,tol=1e-8,  x0=population, method='BFGS', options=opt)
	
	def calculateELA(self):
		sample = self.currentResults.iloc[:,0:self.dimension].values
		obj_values = self.currentResults['y'].values
		featureObj = create_feature_object(sample,obj_values, lower=-5, upper=5)

		try:
			ela_distr = calculate_feature_set(featureObj, 'ela_distr')
		except:
			ela_distr = {}
		
		try:
			ela_level = calculate_feature_set(featureObj, 'ela_level')
		except:
			ela_level = {}

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

	def saveElaFeat(self, name):
		#If Pflacco is bifurcated, it will save a csv of the current results before the local search
		if self.pflacco:
			self.performance.insertELAData(name, self.ela_feat)
		else:
			self.currentResults.to_csv('temp/'+name+'_pflacco.csv',index=False)
	
	def calculatePerformance(self, name):
		optimalValue = self.optimalValue  
		ert8, fce, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue))

		ert7, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-7)))

		ert6, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-6)))

		ert5, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-5)))

		ert4, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-4)))

		ert3, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-3)))

		ert2, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-2)))

		ert1, _, _, _, minValue = self._calcFCEandERT(fitnesses=np.array([list(self.currentResults['y'].values)]),target=(optimalValue-(1e-8)+(1e-1)))
		
		self.performance.insertPerformance(name, ert8, ert7, ert6, ert5, ert4, ert3, ert2, ert1, fce)

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
			:param target:      Target value to use for basing the ERT on. Default: 1e-8
			:return:            ESFitness object with FCE and ERT properly set
			Source: Modea Python Package by https://github.com/sjvrijn, slightly modified to consider target based on BBOB target value.
		"""
		min_fitnesses = np.min(fitnesses, axis=1).tolist()  # Save as list to ensure eval() can read it as summary

		num_runs, num_evals = fitnesses.shape
		below_target = fitnesses < target
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
