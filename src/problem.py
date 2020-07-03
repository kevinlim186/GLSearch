import bbobbenchmarks.bbobbenchmarks as bn
from modea.Utils import getOpts, getVals, options,initializable_parameters
from modea import Algorithms
import numpy as np
from scipy.optimize import minimize
import pandas as pd

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
		self.currentResults =  pd.DataFrame(columns=['x1', 'x2', 'x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14', 'x15', 'x16','x17','x18','x19','x20', 'y'])

		self.problemInstance = None
		self.optimizer = None

		self.createProblemInstance()
		self.initializedESAlgorithm()
	
	def getProblemName(self, functionID, instance, budget, local):
		functionNamesNoiseless = [ 
                    '1 Noise-free Sphere function', 
                    '2 Separable ellipsoid with monotone transformation', 
                    '3 Rastrigin with monotone transformation separable "condition" 10', 
                    '4 skew Rastrigin-Bueche, condition 10, skew-"condition" 100', 
                    '5 Linear slope', 
                    '6 Attractive sector function',
                    '7 Step-ellipsoid, condition 100, noise-free',
                    '8 Rosenbrock noise-free',
                    '9 Rosenbrock, rotated', 
                    '10 Ellipsoid with monotone transformation, condition 1e6', 
                    '11 Discus (tablet) with monotone transformation, condition 1e6', 
                    '12 Bent cigar with asymmetric space distortion, condition 1e6',
                    '13 Sharp ridge',
                    '14 Sum of different powers, between x^2 and x^6, noise-free',
                    '15 Rastrigin with asymmetric non-linear distortion, "condition" 10',
                    '16 Weierstrass, condition 100',
                    '17 Schaffers F7 with asymmetric non-linear transformation, condition 10',
                    '18 Schaffers F7 with asymmetric non-linear transformation, condition 1000',
                    '19 F8F2 sum of Griewank-Rosenbrock 2-D blocks, noise-free',
                    '20 Schwefel with tridiagonal variable transformation',
                    '21 Gallagher with 101 Gaussian peaks, condition up to 1000, one global rotation, noise-free',
                    '22 Gallagher with 21 Gaussian peaks, condition up to 1000, one global rotation',
                    '23 Katsuura function',
                    '24 Lunacek bi-Rastrigin, condition 100',
                    '25 The number of legs the animal (default is 4)'
                    ]
    
		if  not isinstance(instance, int):
			raise ValueError('Please enter a valid instance number')
		
		esConfig = ''.join([str(elem) for elem in self.esconfig if elem is not None]) 
		if instance >0 and instance <25:
			functionIndex = functionID - 1
			functionAttr = functionNamesNoiseless[functionIndex]
			
			functionName = '_F'+str(functionID)+functionAttr + '_I' + str(self.instance) + '_D' + str(self.dimension)+ '_ES'+esConfig + "_B" + str(budget) + "_Local:" + str(local) 
			return functionName
		elif functionID == 0:
			functionAttr = 'Parabola'
			functionName = '_F'+str(functionID)+functionAttr + '_I' + str(self.instance) + '_D' + str(self.dimension)+ '_ES'+esConfig + "_B" + str(budget) + "_Local:" + str(local) 
			return functionName

	def createProblemInstance(self):
		if self.instance >0 and self.instance <25:
			functionAttr = 'F' + str(self.function)
			function = getattr(bn, functionAttr)(self.instance)
			def functionInstance(x):
				self.RemainingBudget = self.TotalBudget - self.RemainingBudget - 1
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
				self.RemainingBudget = self.TotalBudget - self.RemainingBudget - 1
				self.SpentBudget = self.SpentBudget + 1
				result = sum([number**2 for number in x])
				data = {}
				data['y'] = result
				for i in range(len(x)):
					data['x'+str(i+1)] = x[i]
				self.currentResults = self.currentResults.append(data, ignore_index=True,)
				return result
			self.problemInstance = parabola
	
	def evaluateOneGeneration(self):
		self.optimizer.runOneGeneration()
		self.optimizer.recordStatistics()

	
	def runOptimizer(self):
		checkpoints = self.getCheckPoints()
		currentLength = 0
		maxIndex = len(checkpoints) 
		while self.TotalBudget > self.SpentBudget:
			if (checkpoints[currentLength] > self.SpentBudget and currentLength+1 < maxIndex):
				currentLength += 1
				#copy the old budget and results so we can continue the evaluation
				remain = self.RemainingBudget 
				spent = self.SpentBudget 
				result = self.currentResults
				x0 = np.array(self.optimizer.best_individual.genotype.flatten())
				name = self.getProblemName(self.function, self.instance, self.SpentBudget,'nedler')
				self.simplexAlgorithm(x0)

				self.currentResults.to_csv('temp/'+name+'.csv',index=False)
				self.performance.importLocalFile('historicalPath','temp/'+name+'.csv')
				#Upload to database
				'''
				for row in self.currentResults.itertuples(index=False):
					self.performance.insertPathData(name, row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20])
				'''
				
				#Reset the budget counter
				self.RemainingBudget = remain
				self.SpentBudget = spent
				self.currentResults = result
			
			self.optimizer.runOneGeneration()
			self.optimizer.recordStatistics()

		name = self.getProblemName(self.function, self.instance, self.SpentBudget, 'Base')
		
		self.currentResults.to_csv('temp/'+name+'.csv',index=False)
		self.performance.importLocalFile('historicalPath','temp/'+name+'.csv')

		'''
		for row in self.currentResults.itertuples(index=False):
			self.performance.insertPathData(name, row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20])
		'''
		

	def initializedESAlgorithm(self):
		representation = self.ensureFullLengthRepresentation(self.esconfig)
		opts = getOpts(representation[:len(options)])
		values = getVals(representation[len(options)+2:])
		values = getVals(self.esconfig)

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
		checkpoints = range(self.checkPoint, self.TotalBudget, self.checkPoint)
		return checkpoints

	def simplexAlgorithm(self, population):
		maxiter = self.RemainingBudget
		opt={'maxfev': maxiter, 'disp': False, 'return_all': True}

		res = minimize(self.problemInstance, x0=population, method='nelder-mead', options=opt)
