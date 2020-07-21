from src.problem import Problem

class Suites:
	def __init__(self, instances, baseBudget, dimensions, esconfig, function, performance, pflacco ):
		self.esconfig = esconfig
		self.instances = instances
		self.performance = performance
		self.dimensions =  dimensions
		self.baseBudget = baseBudget
		self.function = function
		self.pflacco = pflacco
	

	def runTest(self):
		maxDimensionLen = len(self.dimensions)
		maxInstanceLen = len(self.instances)
		
		for i in range(maxDimensionLen):
			dimension = self.dimensions[i]
			budget = self.baseBudget * self.dimensions[i]
			checkPoint = 500 * self.dimensions[i]
			for j in range(maxInstanceLen):
				instance = self.instances[j]
				self.runProblem(budget=budget, function=self.function, instance=instance, dimension=dimension, esconfig=self.esconfig, checkPoint=checkPoint, logger=self.performance, pflacco=self.pflacco)

	def runProblem (self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco):
			problem = Problem(budget, function, instance, dimension, esconfig, checkPoint, logger,pflacco)
			problem.runTest()
			problem.saveElaFeat('Function_'+str(function))


		