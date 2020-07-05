from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
from modea import  Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
from modea.Utils import ESFitness, getOpts, getVals, reprToString, options, initializable_parameters, ESFitness
import os
import numpy as np
from src.Util import createBBOBFunction, runOneGeneration, initializePopulation, graph2DFunction, performanceLogger,ensureFullLengthRepresentation, computeERT
import pandas as pd
import seaborn as sns 
#Config
n=2
budget=1000  
l_bound=-5
u_bound=5
directory = './results/'

#initiate the function
function = createBBOBFunction(functionID=1, instanceNumber=0)

#graph the function

resolution = 0.1
graph2DFunction(lbound=l_bound, ubound=u_bound,resolution=resolution, function=function['function'], functionName=function['name'], directory = directory+function['name']+'.png')


#Customized ES
#max Values [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
starting_configuration = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
configName =  ''.join([str(elem) for elem in starting_configuration]) 
representation = ensureFullLengthRepresentation(starting_configuration)

# Interpret the representation into parameters for the ES
opts = getOpts(representation[:len(options)])
values = getVals(representation[len(options)+2:])
values = getVals(starting_configuration)

# Initialize the algorithm
customES = Algorithms.CustomizedES(2, function['function'], budget=1000, opts=opts, values=values)
customES.mutateParameters = customES.parameters.adaptCovarianceMatrix

customES.runOneGeneration()
customES.new_population
customES.population
customES.recordStatistics()
customES.used_budget
customES.runOptimizer()
customES.
#Get the ERT 
performanceDirectory = directory+function['name']+' '+ configName + '.csv'
customES, performance = computeERT(customES, target=1e-8, directory=performanceDirectory)

customES.best_individual
#graphing for fitness over time
fitness = pd.DataFrame({'Fitness Over Time': customES.fitness_over_time, 'Sigma Over Time': customES.sigma_over_time})

plt.clf()
plot = sns.lineplot( markers=True, data=fitness, palette="tab10", linewidth=2.5)
plt.title(directory+function['name']+' '+ configName)
plt.savefig(directory+function['name']+' '+ configName + '.png')


from src.problem import Problem 
from src.logger import Performance

performance = Performance()



problem = Problem( 1000, 11, 0, 2, [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], 100,performance)
problem2 = Problem( 1000, 0, 0, 2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2], 100,performance)

problem2.TotalBudget
problem2.RemainingBudget
problem2.SpentBudget
problem.getProblemName(0,0,200,"ce")
problem.runOptimizer()
problem2.runOptimizer()

problem.optimizer.population
problem.optimizer.new_population 
problem.optimizer.used_budget

problem.optimizer.u_bound
problem.optimizer.l_bound
problem.optimizer.population = None
problem.optimizer.initializePopulation()

problem.SpentBudget

name='_F44 skew Rastrigin-Bueche_I0_D2_ES11111111122_B5_Local:nedler'
performance.importLocalFile('historicalPath','temp/'+name+'.csv')

fitness = ESFitness(fitnesses=problem.currentResults['y'].values, target=0.00005)


fitness = ESFitness(fitnesses=np.array([list(problem.currentResults['y'].values)]), target=0.005)
fitness = ESFitness(fitnesses=np.array([problem.optimizer.fitness_over_time]), target=0.005)
data['FCE'] = fitness.FCE
data['ERT'] = fitness.ERT