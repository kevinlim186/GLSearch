import bbobbenchmarks.bbobbenchmarks as bn
from modea.Utils import getOpts, getVals, options,initializable_parameters, ESFitness
from modea import Algorithms, Parameters
import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
from pflacco.pflacco import calculate_feature_set, create_feature_object

class Problem():
	def __init__(self, budget, function, instance, dimension, esconfig, checkPoint, logger):
		self.totalBudget = budget
		self.remainingBudget = budget
		self.spentBudget = 0
		self.function = function
		self.instance = instance
		self.dimension = dimension
		self.esconfig = esconfig 
		self.performance = logger
		self.checkPoint = checkPoint
		self.currentResults =  pd.DataFrame(columns=['x1', 'x2', 'x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14', 'x15', 'x16','x17','x18','x19','x20', 'y', 'name'])
		self.elaFetures =  pd.DataFrame(columns=['name', 'ela_distr', 'ela_level', 'ela_meta', 'basic', 'disp', 'limo', 'nbc', 'pca', 'ic'])
		self.prevRemainingBudget = None
		self.prevSpentBudget = None
		self.prevcurrentResults = None

		self.ela_feat = None

		self.problemInstance = None
		self.optimizer = None
		self.optimalValue = None

		self.createProblemInstance()
		self.initializedESAlgorithm()
	
	def getProblemName(self, functionID, instance, budget, local):
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
			
			functionName = '_F'+functionAttr + '_I' + str(self.instance) + '_D' + str(self.dimension)+ '_ES'+esConfig + "_B" + str(budget) + "_Local:" + str(local) 
			return functionName
		elif functionID == 0:
			functionAttr = 'Parabola'
			functionName = '_F'+str(functionID)+functionAttr + '_I' + str(self.instance) + '_D' + str(self.dimension)+ '_ES'+esConfig + "_B" + str(budget) + "_Local:" + str(local) 
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
	
	
	def runOptimizer(self):
		checkpoints = self.getCheckPoints()
		currentLength = 0
		maxIndex = len(checkpoints) 
		while self.totalBudget > self.spentBudget:
			if (checkpoints[currentLength] < self.spentBudget and currentLength < maxIndex):
				currentLength += 1
				self._printProgressBar(currentLength, maxIndex-1,prefix='Problem with '+str(self.dimension) + 'd - f'+ str(self.function) + ' - i' + str(self.instance),length=50)
				# Get the best individuals as of this time as input to the local search. Calculate the ELA features
				x0 = np.array(self.optimizer.best_individual.genotype.flatten())
				self.calculateELA()

				#Simplex Method
				self.saveState()
				
				name = self.getProblemName(self.function, self.instance, self.spentBudget,'nedler')
				self.saveElaFeat(name)
				self.simplexAlgorithm(x0)

				self.calculatePerformance(name)
				self.currentResults['name'] = name
				self.currentResults.to_csv('temp/'+name+'.csv',index=False)
				self.performance.importHistoricalPath('temp/'+name+'.csv')
				
				self.loadState()
				

				#Gradient Descent Method 0.1
				name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs0.1')
				self.saveElaFeat(name)
				self.bfgsAlgorithm(x0, 0.1)

				self.calculatePerformance(name)
				self.currentResults['name'] = name
				self.currentResults.to_csv('temp/'+name+'.csv',index=False)
				self.performance.importHistoricalPath('temp/'+name+'.csv')
				
				self.loadState()

				#Gradient Descent Method 0.3
				name = self.getProblemName(self.function, self.instance, self.spentBudget,'bfgs0.3')
				self.saveElaFeat(name)
				self.bfgsAlgorithm(x0, 0.3)

				self.calculatePerformance(name)
				self.currentResults['name'] = name
				self.currentResults.to_csv('temp/'+name+'.csv',index=False)
				self.performance.importHistoricalPath('temp/'+name+'.csv')
				
				self.loadState()
			
			self.optimizer.runOneGeneration()
			self.optimizer.recordStatistics()

		name = self.getProblemName(self.function, self.instance, self.spentBudget, 'Base')
		
		self.currentResults['name'] = name
		self.calculatePerformance(name)
		self.currentResults.to_csv('temp/'+name+'.csv',index=False)
		self.performance.importHistoricalPath('temp/'+name+'.csv')


	def saveState(self):
		self.prevRemainingBudget  = self.remainingBudget 
		self.prevSpentBudget  = self.spentBudget 
		self.prevCurrentResults= self.currentResults

	def loadState(self):
		self.remainingBudget = self.prevRemainingBudget
		self.spentBudget = self.prevSpentBudget 
		self.currentResults = self.prevCurrentResults

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
		x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
		opt={'maxfev': maxiter, 'disp': False, 'return_all': False}

		minimize(self.problemInstance,tol=1e-8, x0=population, method='nelder-mead', bounds = x_bounds, options=opt)

	def bfgsAlgorithm(self, population, stepSize):
		maxiter = self.remainingBudget
		x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
		opt={'maxfev': maxiter, 'disp': False, 'return_all': False, 'eps': stepSize}

		minimize(self.problemInstance,tol=1e-8, x0=population, method='BFGS', bounds = x_bounds, options=opt)
	
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
		self.performance.insertELAData(name, self.ela_feat)
	
	def calculatePerformance(self, name):
		fitness = ESFitness(fitnesses=np.array([list(self.currentResults['y'].values)]), target=self.optimalValue)
		ert = fitness.ERT
		fce = fitness.FCE
		self.performance.insertPerformance(name, ert, fce)
	
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