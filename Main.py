from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
import bbobbenchmarks.bbobbenchmarks as bn
from modea import Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
from modea.Utils import getOpts, getVals, reprToString, options, initializable_parameters, ESFitness
import os
import numpy as np
from functools import partial
from Util import createBBOBFunction, runOneGeneration, initializePopulation, graph2DFunction
import pandas as pd
import seaborn as sns

#Config
n=2
budget=1000
l_bound=-5
u_bound=5
mu=20
lambda_=20

#initiate the function
function = createBBOBFunction(functionID=0, instanceNumber=44)

#graph the function
resolution = 0.1
graph2DFunction(lbound=l_bound, ubound=u_bound,resolution=resolution, function=function['function'])


# We use functions here to 'hide' the additional passing of parameters that are algorithm specific
parameters = Parameters.Parameters(n=n, budget=budget,l_bound=l_bound, u_bound=u_bound, mu=mu, lambda_=lambda_)
recombine = Recombination.random
mutate = partial(Mutation.CMAMutation, sampler=Sampling.GaussianSampling(2))
select = Selection.onePlusOneSelection
mutateParameters = parameters.oneFifthRule
population = initializePopulation(parameters)


functions = {
    'recombine': recombine,
    'mutate': mutate,
    'select': select,
    'mutateParameters': mutateParameters,
}

#EvolutionaryOptimizer (OK)
evolutionaryOptimizer = Algorithms.EvolutionaryOptimizer(population, function['function'], budget=parameters.budget, functions= functions,parameters=parameters)

generation = runOneGeneration(evolutionaryOptimizer)
generation['EvolutionaryOptimizer']
generation['Logger']
generation['FeatureObj']
generation['EvolutionaryOptimizer'].population

evolutionaryOptimizer.runOptimizer()

evolutionaryOptimizer.fitness_over_time
evolutionaryOptimizer.sigma_over_time


# Calculate all features
ela_features = calculate_features(generation['FeatureObj'])
print(ela_features)


#Computing for fitness over time
fitness = ESFitness(fitnesses=np.array([evolutionaryOptimizer.fitness_over_time]), target=1.565)
fitness.FCE
fitness.ERT

#graphing for fitness over time
fitness = pd.DataFrame({'Fitness Over Time': evolutionaryOptimizer.fitness_over_time, 'Sigma Over Time': evolutionaryOptimizer.sigma_over_time})
    
plot = sns.lineplot( markers=True, data=fitness, palette="tab10", linewidth=2.5)
plt.show()




# Arbitrary objective function
def objective_function(x, dim):
    return [entry[0]**2 - entry[1]**2 for entry in x]


# Create inital sample using latin hyper cube sampling
sample = create_initial_sample(100, 2, type = 'lhs')
# Calculate the objective values of the initial sample using an arbitrary objective function (here y = x1^2 - x2^2)
obj_values = objective_function(sample, 2)
# Create feature object
feat_object = create_feature_object(sample, obj_values, lower=-10,upper=10, blocks=3)

# Calculate a single feature set
cm_angle_features = calculate_feature_set(feat_object, 'cm_angle')


# Calculate all features
ela_features = calculate_features(feat_object)
ela_features['gcm.near.basin_prob.sd']
