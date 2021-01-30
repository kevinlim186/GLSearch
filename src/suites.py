from src.problem import Problem
import math
from numpy import log

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
            problem.runDataGathering(size= [50,100,200, '5G', '50G', '100G'])
            #problem.saveElaFeat('Function_'+str(function))

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

    def runTestModel(self,  ASP, size,restart, features, ASPName, stepSize):
        maxDimensionLen = len(self.dimensions)
        maxInstanceLen = len(self.instances)
        
        for i in range(maxDimensionLen):
            dimension = self.dimensions[i]
            budget = self.baseBudget * self.dimensions[i]
            checkPoint = 500 * self.dimensions[i]
            for j in range(maxInstanceLen):
                instance = self.instances[j]
                self.runModelTest(budget=budget, function=self.function, instance=instance, dimension=dimension, esconfig=self.esconfig, checkPoint=checkPoint, logger=self.performance, pflacco=self.pflacco, localSearch=self.localSearch,  ASP=ASP, size=size,restart=restart, features=features, ASPName=ASPName, stepSize=stepSize)


    def runModelTest(self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco,localSearch, ASP, size,restart, features, ASPName,stepSize):
            problem = Problem(budget, function, instance, dimension, esconfig, checkPoint, logger,pflacco,localSearch)
            problem.runASPBattery(ASP, ASPName, size,stepSize,restart, features)
        

    def runTestMultipleModel(self,  models, stepSize, precision):
        maxDimensionLen = len(self.dimensions)
        maxInstanceLen = len(self.instances)
        
        for i in range(maxDimensionLen):
            dimension = self.dimensions[i]
            budget = self.baseBudget * self.dimensions[i]
            checkPoint = 500 * self.dimensions[i]
            for j in range(maxInstanceLen):
                instance = self.instances[j]
                self.runModelTestMultiple(budget=budget, function=self.function, instance=instance, dimension=dimension, esconfig=self.esconfig, checkPoint=checkPoint, logger=self.performance, pflacco=self.pflacco, localSearch=self.localSearch,  models=models, stepSize=stepSize, precision=precision)


    def runModelTestMultiple(self, budget, function, instance, dimension, esconfig, checkPoint, logger, pflacco,localSearch,models,stepSize, precision):
            problem = Problem(budget, function, instance, dimension, esconfig, checkPoint, logger,pflacco,localSearch, precision=precision)
            problem.runTestMultipleModels(models,  stepSize)