from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
import bbobbenchmarks.bbobbenchmarks as bn
from modea import Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
from modea.Utils import getOpts, getVals, reprToString, options, initializable_parameters
import os
import numpy as np
from functools import partial
import math
from Util import createBBOBFunction

#Test function where we know the outcome. Zero is the best function
def test_function(x):
    return sum([-(number**2) for number in x])


# objective function
def objective_function(x, dim):
    return [entry[0]**2 - entry[1]**2 for entry in x]

# Create inital sample using latin hyper cube sampling
sample = create_initial_sample(100, 2, type = 'lhs')
# Calculate the objective values of the initial sample using an arbitrary objective function (here y = x1^2 - x2^2)
obj_values = objective_function(sample, 2)
# Create feature object
feat_object = create_feature_object(sample, obj_values, blocks=3)

# Calculate a single feature set
cm_angle_features = calculate_feature_set(feat_object, 'cm_angle')
print(cm_angle_features)

# Calculate all features
ela_features = calculate_features(feat_object)
print(ela_features)






# We use functions here to 'hide' the additional passing of parameters that are algorithm specific
parameters = Parameters.Parameters(n=2, budget=100,l_bound=5, u_bound=100)
recombine = Recombination.onePlusOne
mutate = partial(Mutation.CMAMutation, sampler=Sampling.GaussianSampling(2))
select = Selection.onePlusOneSelection
mutateParameters = parameters.oneFifthRule

functions = {
    'recombine': recombine,
    'mutate': mutate,
    'select': select,
    'mutateParameters': mutateParameters,
}

population = [Individual.FloatIndividual(2)]



#CMAESOptimizer
cMAESOptimizer = Algorithms.CMAESOptimizer(2,fitnessFunction=test_function, budget=100, mu=2, lambda_=3, elitist=False)

parameters = Parameters.Parameters(n, budget, mu, lambda_, elitist=elitist)
population = [Individual.FloatIndividual(n) for _ in range(parameters.mu_int)]
pop = population

parameters = Parameters.Parameters(n=2, budget=100,mu=2, lambda_=3)
wcm = parameters.wcm
parameters.weights

for individual in population:
    individual.genotype = wcm

offspring = np.column_stack([ind.genotype for ind in pop])
param.offspring = offspring

param.wcm = dot(offspring, parameters.weights)

param.wcm = dot(np.squeeze(np.asarray(offspring)), np.squeeze(np.asarray(parameters.weights)))



#CustomizedES
#Dictionary of options
option = {
    'active': True,
    'elitist': True, #Current best will not be killed and allow to persist
    'mirrored': True,
    'orthogonal': True, 
    'sequential': True,
    'threshold': True,
    'tpa': True,
    'base-sampler': 'quasi-sobol', #choose between 'quasi-sobol' or 'quasi-halton' or None
    'ipop': 'IPOP', #choose between 'IPOP' or 'BIPOP' or None
    'selection': 'pairwise', # choose between 'pairwise' or None
    'weights_option': '1/n' # choose between '1/n'
    }

values = [0,0,0,0,0,0,0,0,0,0,0]

customizedES = Algorithms.CustomizedES(n=2,fitnessFunction=test_function,budget=100,mu=2,lambda_=4,opts=option, values=values)
customizedES.runOneGeneration()




# Configuration of the 11 Modules
starting_configuration = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
representation = ensureFullLengthRepresentation(starting_configuration)
# Interpret the representation into parameters for the ES
opts = getOpts(representation[:len(options)])
lambda_ = representation[len(options)]
mu = representation[len(options)+1]
values = getVals(representation[len(options)+2:])
ES_BUDGET_FACTOR = 10
ndim = 2
NUMBER_OF_RECONFIGURATION=1
# Init CMA-Es
custom_es = Algorithms.CustomizedES(n = ndim,
                            fitnessFunction = test_function,
                            budget = math.floor((ES_BUDGET_FACTOR * ndim) / NUMBER_OF_RECONFIGURATION),
                            mu = mu,
                            lambda_ = lambda_,
                            opts = opts,  #Configuration of Modules
                            values = values)

custom_es.mutateParameters = custom_es.parameters.adaptCovarianceMatrix

generation_size, sigma_over_time, fitness_over_time, best_individual = Algorithms._customizedES(n = ndim, fitnessFunction = test_function, budget = math.floor((ES_BUDGET_FACTOR * ndim) / NUMBER_OF_RECONFIGURATION), mu = mu, lambda_ = lambda_, opts = opts, values = values)



# Run CMA-ES for one generation
custom_es.runOneGeneration()
custom_es.recordStatistics()
custom_es.total_used_budget


custom_es.runOptimizer()
custom_es.runLocalRestartOptimizer()
custom_es.best_individual
custom_es.total_budget
custom_es.total_used_budget




#EvolutionaryOptimizer (OK)
evolutionaryOptimizer = Algorithms.EvolutionaryOptimizer(population, test_function, 100,  functions= functions,parameters=parameters)

evolutionaryOptimizer.runOptimizer()
evolutionaryOptimizer.runLocalRestartOptimizer()

evolutionaryOptimizer.total_budget
evolutionaryOptimizer.total_used_budget


#GA Opt
gAOptimizer = Algorithms.GAOptimizer(2, test_function, 100, mu=5, lambda_=10, population = None,parameters=parameters)
gAOptimizer.initializePopulation()
gAOptimizer.evalPopulationSequentially()
gAOptimizer.runOneGeneration()
gAOptimizer.runOptimizer()

#General Flow of the MIESOptimizer 
mies = Individual.MixedIntIndividual(n=2,num_discrete=1,num_floats=0, num_ints=1)
mies.genotype = [100,100]
population2 = [mies,mies]

mIESOptimizer = Algorithms.MIESOptimizer(n=2, mu=2, lambda_=2, population=population2, fitnessFunction=test_function,budget=100)

mIESOptimizer.runOptimizer()


#General Flow of the One plus one optimizer (OK)
onePlustOneOptimizer = Algorithms.OnePlusOneOptimizer(2, test_function, 100)
onePlustOneOptimizer.total_used_budget
onePlustOneOptimizer.fitness_over_time
onePlustOneOptimizer.best_individual
onePlustOneOptimizer.initializePopulation()
onePlustOneOptimizer.runOptimizer()
onePlustOneOptimizer.evalPopulationSequentially()


individual = Individual.FloatIndividual(2) #individual with 2 dimensions
Mutation.adaptStepSize(individual)