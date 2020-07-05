import bbobbenchmarks.bbobbenchmarks as bn
from modea.Utils import getOpts, getVals, options,initializable_parameters, ESFitness
from modea import Algorithms, Parameters
import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
from pflacco.pflacco import calculate_feature_set, create_feature_object

class Problem():
	def __init__(self, budget, function, instance, dimension, esconfig, checkPoint, logger):
		self.TotalBudget = budget
		self.RemainingBudget = budget
		self.SpentBudget = 0
		self.function = function
		self.instance = instance
		self.dimension = dimension
		self.esconfig = esconfig 
		self.performance = logger
		self.checkPoint = checkPoint
		self.currentResults =  pd.DataFrame(columns=['x1', 'x2', 'x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14', 'x15', 'x16','x17','x18','x19','x20', 'y', 'name'])
		self.elaFetures =  pd.DataFrame(columns=['name', 'ela_distr', 'ela_level', 'ela_meta', 'basic', 'disp', 'limo', 'nbc', 'pca', 'ic'])

		self.problemInstance = None
		self.optimizer = None

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
				self.RemainingBudget = self.RemainingBudget - 1
				self.SpentBudget = self.SpentBudget + 1
				result = function(x)
	
				data = {}
				data['y'] = result
				for i in range(len(x)):
					data['x'+str(i+1)] = x[i]
				self.currentResults = self.currentResults.append(data, ignore_index=True,)
				return result
			self.problemInstance = functionInstance
		elif self.function == 0:
			def parabola(x):
				self.RemainingBudget = self.RemainingBudget - 1
				self.SpentBudget = self.SpentBudget + 1
				result = sum([number**2 for number in x])
				data = {}
				data['y'] = result
				for i in range(len(x)):
					data['x'+str(i+1)] = x[i]
				self.currentResults = self.currentResults.append(data, ignore_index=True,)
				return result
			self.problemInstance = parabola
	
	
	def runOptimizer(self):
		checkpoints = self.getCheckPoints()
		currentLength = 0
		maxIndex = len(checkpoints) 
		while self.TotalBudget > self.SpentBudget:
			if (checkpoints[currentLength] < self.SpentBudget and currentLength < maxIndex):
				currentLength += 1
				#copy the old budget and results so we can continue the evaluation
				remain = self.RemainingBudget 
				spent = self.SpentBudget 
				result = self.currentResults
				x0 = np.array(self.optimizer.best_individual.genotype.flatten())
				name = self.getProblemName(self.function, self.instance, self.SpentBudget,'nedler')
				self.calculateELA(name)
				self.simplexAlgorithm(x0)

				self.calculatePerformance(name)
				self.currentResults['name'] = name
				self.currentResults.to_csv('temp/'+name+'.csv',index=False)
				self.performance.importHistoricalPath('temp/'+name+'.csv')
				
				#Reset the budget counter
				self.RemainingBudget = remain
				self.SpentBudget = spent
				self.currentResults = result
			
			self.optimizer.runOneGeneration()
			self.optimizer.recordStatistics()

		name = self.getProblemName(self.function, self.instance, self.SpentBudget, 'Base')
		
		self.currentResults['name'] = name
		self.calculatePerformance(name)
		self.currentResults.to_csv('temp/'+name+'.csv',index=False)
		self.performance.importHistoricalPath('temp/'+name+'.csv')

		

	def initializedESAlgorithm(self):
		representation = self.ensureFullLengthRepresentation(self.esconfig)
		opts = getOpts(representation[:len(options)])
		values = getVals(representation[len(options)+2:])
		values = getVals(self.esconfig)


		parameters = Parameters.Parameters(n=5,budget=1000,  l_bound=-5, u_bound=5)

		customES = Algorithms.CustomizedES(self.dimension, self.problemInstance, budget=self.TotalBudget, opts=opts, values=values)

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
		checkpoints = range(1, self.TotalBudget + self.checkPoint , self.checkPoint)
		return checkpoints

	def simplexAlgorithm(self, population):
		maxiter = self.RemainingBudget
		x_bounds = Bounds(np.array([-5.]), np.array([5.]), keep_feasible = True)
		print(maxiter)
		opt={'maxfev': maxiter, 'disp': False, 'return_all': False}

		res = minimize(self.problemInstance,tol=1e-8, x0=population, method='nelder-mead', bounds = x_bounds, options=opt)
	
	def calculateELA(self, name):
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

		ela_feat =  {**ela_distr, **ela_level, **ela_meta, **basic, **disp, **limo, **nbc, **pca, **ic }

		self.performance.insertELAData(name, ela_feat)
	
	def calculatePerformance(self, name):
		fitness = ESFitness(fitnesses=np.array([list(self.currentResults['y'].values)]), target=1e-8)
		ert = fitness.ERT
		fce = fitness.FCE
		self.performance.insertPerformance(name, ert, fce)