from src.problem import Problem

class Suites:
    def __init__(self, instances, baseBudget, dimensions, esconfig, function, performance, pflacco, localSearch=None ):
        self.esconfig = esconfig
        self.instances = instances
        self.performance = performance
        self.dimensions =  dimensions
        self.baseBudget = baseBudget
        self.function = function
        self.pflacco = pflacco
        self.localSearch = localSearch
    

    def runDataGathering(self):
        maxDimensionLen = len(self.dimensions)
        maxInstanceLen = len(self.instances)
        
        for i in range(maxDimensionLen):
            dimension = self.dimensions[i]
            budget = self.baseBudget * self.dimensions[i]
            checkPoint = 500 * self.dimensions[i]
            for j in range(maxInstanceLen):
                instance = self.instances[j]
                self.runProblem(budget=budget, function=self.function, instance=instance, dimension=dimension, esconfig=self.esconfig, checkPoint=checkPoint, logger=self.performance, pflacco=self.pflacco,localSearch= self.localSearch)

    def runProblem (self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco, localSearch):
            problem = Problem(budget, function, instance, dimension, esconfig, checkPoint, logger,pflacco, localSearch)
            problem.runDataGathering()
            problem.saveElaFeat('Function_'+str(function))

    def runTestSuite(self):
        maxDimensionLen = len(self.dimensions)
        maxInstanceLen = len(self.instances)
        
        for i in range(maxDimensionLen):
            dimension = self.dimensions[i]
            budget = self.baseBudget * self.dimensions[i]
            checkPoint = 500 * self.dimensions[i]
            for j in range(maxInstanceLen):
                instance = self.instances[j]
                self.runTest(budget=budget, function=self.function, instance=instance, dimension=dimension, esconfig=self.esconfig, checkPoint=checkPoint, logger=self.performance, pflacco=self.pflacco, localSearch=self.localSearch)


    def runTest(self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco,localSearch):
            problem = Problem(budget, function, instance, dimension, esconfig, checkPoint, logger,pflacco,localSearch)
            problem.runTest()

    def runTestModel(self,  ASP, size,restart):
        maxDimensionLen = len(self.dimensions)
        maxInstanceLen = len(self.instances)
        
        for i in range(maxDimensionLen):
            dimension = self.dimensions[i]
            budget = self.baseBudget * self.dimensions[i]
            checkPoint = 500 * self.dimensions[i]
            for j in range(maxInstanceLen):
                instance = self.instances[j]
                self.runModelTest(budget=budget, function=self.function, instance=instance, dimension=dimension, esconfig=self.esconfig, checkPoint=checkPoint, logger=self.performance, pflacco=self.pflacco, localSearch=self.localSearch,  ASP=ASP, size=size,restart=restart)


    def runModelTest(self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco,localSearch, ASP, size,restart):
            problem = Problem(budget, function, instance, dimension, esconfig, checkPoint, logger,pflacco,localSearch)
            problem.runASPBattery(ASP, size,restart)
        