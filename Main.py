from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
from modea import Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
from modea.Utils import getOpts, getVals, reprToString, options, initializable_parameters, ESFitness
import os
import numpy as np
from Util import createBBOBFunction, runOneGeneration, initializePopulation, graph2DFunction, performanceLogger,ensureFullLengthRepresentation, computeERT
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Config
n=2
budget=1000
l_bound=-5
u_bound=5
directory = './results/'

#initiate the function
function = createBBOBFunction(functionID=1, instanceNumber=44)

#graph the function

resolution = 0.1
graph2DFunction(lbound=l_bound, ubound=u_bound,resolution=resolution, function=function['function'], functionName=function['name'], directory = directory+function['name']+'.png')


#Customized ES
starting_configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
configName =  ''.join([str(elem) for elem in starting_configuration]) 
representation = ensureFullLengthRepresentation(starting_configuration)

# Interpret the representation into parameters for the ES
opts = getOpts(representation[:len(options)])
values = getVals(representation[len(options)+2:])
values = getVals(starting_configuration)

# Initialize the algorithm
customES = Algorithms.CustomizedES(2, function['function'], budget=1000, opts=opts, values=values)
customES.mutateParameters = customES.parameters.adaptCovarianceMatrix
 
#Get the ERT
performanceDirectory = directory+function['name']+' '+ configName + '.csv'
customES, performance = computeERT(customES, target=1, directory=performanceDirectory)

#graphing for fitness over time
fitness = pd.DataFrame({'Fitness Over Time': customES.fitness_over_time, 'Sigma Over Time': customES.sigma_over_time})

plt.clf()
plot = sns.lineplot( markers=True, data=fitness, palette="tab10", linewidth=2.5)
plt.title(directory+function['name']+' '+ configName)
plt.savefig(directory+function['name']+' '+ configName + '.png')

