from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
import bbobbenchmarks.bbobbenchmarks as bn
from modea import Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
from modea.Utils import getOpts, getVals, reprToString, options, initializable_parameters
import os
import numpy as np
from functools import partial
import math
from Util import createBBOBFunction, runOneGeneration


# Create inital sample using latin hyper cube sampling
sample = create_initial_sample(2, 2, type = 'lhs')
# Calculate the objective values of the initial sample using an arbitrary objective function (here y = x1^2 - x2^2)
obj_values = function['batchFunction'](sample)
# Create feature object
feat_object = create_feature_object(sample, obj_values, blocks=3)

# Calculate a single feature set
cm_angle_features = calculate_feature_set(feat_object, 'cm_angle')
print(cm_angle_features)

# Calculate all features
ela_features = calculate_features(feat_object)
print(ela_features)


function = createBBOBFunction(functionID=0, instanceNumber=44)

sample = [[1,1], [2,2], [1,2]]
function['function']([1,2])
function['batchFunction'](sample)

# We use functions here to 'hide' the additional passing of parameters that are algorithm specific
parameters = Parameters.Parameters(n=2, budget=100,l_bound=-5, u_bound=5, mu=5, lambda_=5)
recombine = Recombination.onePlusOne
mutate = partial(Mutation.CMAMutation, sampler=Sampling.GaussianSampling(2))
select = Selection.onePlusOneSelection
mutateParameters = parameters.oneFifthRule
individual = Individual.FloatIndividual(2)
population = [individual,individual,individual,individual,individual]

functions = {
    'recombine': recombine,
    'mutate': mutate,
    'select': select,
    'mutateParameters': mutateParameters,
}

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

#EvolutionaryOptimizer (OK)
evolutionaryOptimizer = Algorithms.EvolutionaryOptimizer(population, function['function'], 100,  functions= functions,parameters=parameters)

len(evolutionaryOptimizer.population)
evolutionaryOptimizer.population [0].genotype
evolutionaryOptimizer.runOneGeneration()


evolutionaryOptimizer.used_budget

generation = runOneGeneration(evolutionaryOptimizer)
generation['EvolutionaryOptimizer']
generation['Logger']
generation['FeatureObj']
generation['EvolutionaryOptimizer'].population


generation = runOneGeneration(generation['EvolutionaryOptimizer'], generation['Logger'])
generation['EvolutionaryOptimizer']
generation['Logger']
generation['FeatureObj']




