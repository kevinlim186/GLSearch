from pflacco.pflacco import *
import bbobbenchmarks.bbobbenchmarks as bn
from modea import Algorithms
import os


# objective function
def objective_function(x):
    return [entry[0]**2 - entry[1]**2 for entry in x]

f3 = bn.F3(13)

optim = Algorithms.EvolutionaryOptimizer(2, f3, 100)
gensize, sigmas, fitness, best_ind = _onePlusOneES(2, sphere, 250)

optim.best_individual
optim.generation_size
optim.fitness_over_time

f3([0, 1, 2])

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








 # short-cut for f3.evaluate([0, 1, 2])


>>> print(bn.instantiate(13)[0])  # returns function instance and optimal f-value
51.53
>>> print bn.nfreeIDs # list noise-free functions
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
>>> for i in bn.nfreeIDs: # evaluate all noiseless functions once
...    print bn.instantiate(i)[0]([0., 0., 0., 0.]),
-77.27454592 6180022.82173 92.9877507529 92.9877507529 140.510117618 70877.9554128 -72.5505202195 33355.7924722 -339.94 4374717.49343 15631566.3487 4715481.0865 550.599783901 -17.2991756229 27.3633128519 -227.827833529 -24.3305918781 131.420159348 40.7103737427 6160.81782924 376.746889545 107.830426761 220.482266557 106.094767386
