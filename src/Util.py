
import bbobbenchmarks.bbobbenchmarks as bn
from modea.Utils import getOpts, getVals, reprToString, options, initializable_parameters, ESFitness
from pflacco.pflacco import create_feature_object
import numpy as np
from pflacco.pflacco import create_initial_sample, calculate_feature_set
from modea.Individual import FloatIndividual
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#from pflacco.pflacco import create_initial_sample , calculate_feature_set, calculate_features
#from modea import Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
#import os
#from functools import partial
#import math

def ensureFullLengthRepresentation(representation):
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

def createBBOBFunction(functionID, instanceNumber):
    '''
    Parameters
        ----------
        FunctionID : int

        Noiseless Function
            0: Parabola (As test function)
            1: Noise-free Sphere function
            2: Separable ellipsoid with monotone transformation
            3: Rastrigin with monotone transformation separable "condition" 10
            4: skew Rastrigin-Bueche, condition 10, skew-"condition" 100
            5: Linear slope
            6: Attractive sector function
            7: Step-ellipsoid, condition 100, noise-free
            8: Rosenbrock noise-free
            9: Rosenbrock, rotated
            10: Ellipsoid with monotone transformation, condition 1e6
            11: Discus (tablet) with monotone transformation, condition 1e6
            12: Bent cigar with asymmetric space distortion, condition 1e6
            13: Sharp ridge
            14: Sum of different powers, between x^2 and x^6, noise-free
            15: Rastrigin with asymmetric non-linear distortion, "condition" 10
            16: Weierstrass, condition 100
            17: Schaffers F7 with asymmetric non-linear transformation, condition 10
            18: Schaffers F7 with asymmetric non-linear transformation, condition 1000
            19: F8F2 sum of Griewank-Rosenbrock 2-D blocks, noise-free
            20: Schwefel with tridiagonal variable transformation
            21: Gallagher with 101 Gaussian peaks, condition up to 1000, one global rotation, noise-free
            22: Gallagher with 21 Gaussian peaks, condition up to 1000, one global rotation
            23: Katsuura function
            24: Lunacek bi-Rastrigin, condition 100 
            The number of legs the animal (default is 4)

        Noisy Function
            101. Sphere with moderate Gaussian noise
            102. Sphere with moderate uniform noise
            103. Sphere with moderate seldom Cauchy noise
            104. Rosenbrock with moderate Gaussian noise
            105. Rosenbrock with moderate uniform noise
            106. Rosenbrock with moderate seldom Cauchy noise
            107. Sphere with Gaussian noise
            108. Sphere with uniform noise
            109. Sphere with seldom Cauchy noise
            110. Rosenbrock with Gaussian noise
            111. Rosenbrock with uniform noise
            112. Rosenbrock with seldom Cauchy noise
            113. Step ellipsoid with Gaussian noise
            114. Step ellipsoid with uniform noise
            115. Step ellipsoid with seldom Cauchy noise
            116. Ellipsoid with Gaussian noise
            117. Ellipsoid with uniform noise
            118. Ellipsoid with seldom Cauchy noise
            119. Different Powers with Gaussian noise
            120. Different Powers with uniform noise
            121. Different Powers with seldom Cauchy noise
            122. Schaffer's F7 with Gaussian noise
            123. Schaffer's F7 with uniform noise
            124. Schaffer's F7 with seldom Cauchy noise
            125. Composite Griewank-Rosenbrock with Gaussian noise
            126. Composite Griewank-Rosenbrock with uniform noise
            127. Composite Griewank-Rosenbrock with seldom Cauchy noise
            128. Gallagher's Gaussian Peaks 101-me with Gaussian noise
            129. Gallagher's Gaussian Peaks 101-me with uniform noise
            130. Gallagher's Gaussian Peaks 101-me with seldom Cauchy noise

        Instance ID : int

    Usage
        ----------
        function = createBBOBFunction(functionID=1, instanceNumber=2)
        function['name'] # Returns the name of the function.
        function['function'] # Returns a function 
        function['function']([1,2,3]) # Evaluates the function based on the given parameters
        function['batchFunction'] # Returns a function that can evaluate batch of inputs
        function['function']([[1,2],[2,3],[3,4]]) # Evaluates the function given batch inputs
    Return
        ----------
        Returns a dictionary with the name, function and batch function as keys
    Source
        ----------
        Functions created here comes from http://coco.lri.fr/backup/COCOdoc/index.html# adapted to work in Python 3.
    '''

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
    
    functionNamesNoisy = [
                    '101 Sphere with moderate Gaussian noise',
                    '102 Sphere with moderate uniform noise',
                    '103 Sphere with moderate seldom Cauchy noise', 
                    '104 Rosenbrock with moderate Gaussian noise', 
                    '105 Rosenbrock with moderate uniform noise',
                    '106 Rosenbrock with moderate seldom Cauchy noise',
                    '107 Sphere with Gaussian noise',
                    '108 Sphere with uniform noise',
                    '109 Sphere with seldom Cauchy noise',
                    '110 Rosenbrock with Gaussian noise',
                    '111 Rosenbrock with uniform noise',
                    '112 Rosenbrock with seldom Cauchy noise',
                    '113 Step ellipsoid with Gaussian noise',
                    '114 Step ellipsoid with uniform noise',
                    '115 Step ellipsoid with seldom Cauchy noise',
                    '116 Ellipsoid with Gaussian noise',
                    '117 Ellipsoid with uniform noise',
                    '118 Ellipsoid with seldom Cauchy noise',
                    '119 Different Powers with Gaussian noise',
                    '120 Different Powers with uniform noise',
                    '121 Different Powers with seldom Cauchy noise',
                    '122 Schaffer\'s F7 with Gaussian noise',
                    '123 Schaffer\'s F7 with uniform noise',
                    '124 Schaffer\'s F7 with seldom Cauchy noise',
                    '125 Composite Griewank-Rosenbrock with Gaussian noise',
                    '126 Composite Griewank-Rosenbrock with uniform noise',
                    '127 Composite Griewank-Rosenbrock with seldom Cauchy noise',
                    '128 Gallagher\'s Gaussian Peaks 101-me with Gaussian noise',
                    '129 Gallagher\'s Gaussian Peaks 101-me with uniform noise',
                    '130 Gallagher\'s Gaussian Peaks 101-me with seldom Cauchy noise'
            ]
    if  not isinstance(instanceNumber, int):
        raise ValueError('Please enter a valid instance number')

    if functionID >0 and functionID <=24:
        functionIndex = functionID - 1
        functionName = functionNamesNoiseless[functionIndex]
        functionAttr = 'F' + str(functionID)
        function = getattr(bn, functionAttr)(instanceNumber)

        def batchFunction(inputValues):
            return [function(entry) for entry in inputValues]


        returnValue = {}
        returnValue['name'] = functionName
        returnValue['function'] = function
        returnValue['batchFunction'] = batchFunction
        return returnValue
    elif functionID >=101 and functionID <=130:
        functionIndex = functionID - 101
        functionName = functionNamesNoisy[functionIndex]
        functionAttr = 'F' + str(functionID)
        function = getattr(bn, functionAttr)(instanceNumber)

        def batchFunction(inputValues):
            return [function(entry) for entry in inputValues]


        returnValue = {}
        returnValue['name'] = functionName
        returnValue['function'] = function
        returnValue['batchFunction'] = batchFunction
        return returnValue
    elif functionID == 0:
        functionName = 'Parabola'
        def parabola(x):
            return sum([number**2 for number in x])
        
        def batchFunction(inputValues):
            return [parabola(entry) for entry in inputValues]
        
        returnValue = {}
        returnValue['name'] = functionName
        returnValue['function'] = parabola
        returnValue['batchFunction'] = batchFunction
        return returnValue
    else:
        raise ValueError("Please choose between Function 1 to 24 for noiseless functions and Function 101 to 130 for noisy function")


def runOneGeneration(EvolutionaryOptimizer, Logger=None):
    '''
    Parameters
        ----------
        EvolutionaryOptimizer: Modea EvolutionaryOptimizer object
        Logger: Array of input and output of previous runs. Leave blank upon initiatization.
    
    Usage
        ----------
        generation = runOneGeneration(evolutionaryOptimizer) # returns a dictionary object
        generation['EvolutionaryOptimizer'] #Modea EvolutionaryOptimizer object after one generation
        generation['Logger'] #Array of inputs and outputs for tracking
        generation['FeatureObj'] #Feature Object of Pflacco
    
    Return
        ----------
        Returns a dictionary with the name, function and batch function as keys

    Source
        ----------
        Implementation of PFlacco and Modea
    '''
    EvolutionaryOptimizer.runOneGeneration()
    generationSize = len(EvolutionaryOptimizer.population)
    EvolutionaryOptimizer.total_used_budget  += generationSize
    EvolutionaryOptimizer.recordStatistics()


    for i in range(generationSize):
        if Logger is None:
            Logger = {}
            Logger['input'] = np.array([EvolutionaryOptimizer.population[i].genotype.flatten().tolist()])
            Logger['output'] = np.array(EvolutionaryOptimizer.population[i].fitness)
        else:
            Logger['input'] = np.append(Logger['input'], np.array([EvolutionaryOptimizer.population[i].genotype.flatten().tolist()]), axis=0)
            Logger['output'] = np.append(Logger['output'], np.array(EvolutionaryOptimizer.population[i].fitness))

    # LHS was used so the number of blocks 
    dim = len(Logger['input'][0])
    sampleSize = len(Logger['input'])
    block = sampleSize / dim

    lowerBound = float(EvolutionaryOptimizer.parameters.l_bound[0][0])
    upperBound = float(EvolutionaryOptimizer.parameters.u_bound[0][0])


    FeatureObj = create_feature_object(x=Logger['input'], y=list(Logger['output']),minimize=True,lower=lowerBound, upper=upperBound,blocks=block)
    
    returnValue = {}
    returnValue['EvolutionaryOptimizer'] = EvolutionaryOptimizer
    returnValue['Logger'] = Logger
    returnValue['FeatureObj'] = FeatureObj

    return returnValue


def initializePopulation(parameters):
    '''
    Parameters
        ----------
        parameter: Modea  Parameters Object
    
    Return
        ----------
        Returns Population List consisting of individuals of type Modea Float Individual

    Source
        ----------
        Implementation of PFlacco and Modea
    '''
    low = parameters.l_bound
    high = parameters.u_bound
    dim = parameters.n
    sampleSize = parameters.mu_int

    lower_bound = [low] * dim
    upper_bound = [high] * dim

    

    sample = create_initial_sample(n_obs=sampleSize,dim=dim,type='lhs',lower_bound=lower_bound,upper_bound=upper_bound)

    population = []

    for genotype in sample:
        individual = FloatIndividual(dim)
        individual.genotype = genotype.reshape(dim,1)
        population.append(individual)

    return population


def graph2DFunction(lbound, ubound, resolution, function, functionName = '',  directory=None):
    '''
    Parameters
        ----------
        lbound: The lower bound value
        ubound: The upper bound value
        resolution: How small the spaces between the list of values
        function: 2D Function to be evaluated (BBOB object)
    
    Return
        ----------
        Graphs the function given the X and Y and Resolution

    Source
        ----------
        Implementation of PFlacco and Modea
    '''

    #Convert BBOB function to act like a normal function 
    def function_n(x,y):
        return function([x,y])

    x = np.arange(lbound, ubound, resolution)
    y = np.arange(lbound,ubound, resolution)
    xx, yy = np.meshgrid(x, y, sparse=True) 
    function_v = np.vectorize(function_n)
    Z = function_v(xx,yy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(functionName)

    Axes3D.plot_wireframe(ax,X=xx, Y=yy,Z=Z)

    if directory is None:
        plt.show()
    else:
        plt.savefig(directory)


def performanceLogger(featureObj, EvolutionaryOptimizer, target, PerformanceLogger=None):
    # weird why is this only 14-- ". In its default settings, flacco computes more than 300 different numerical landscape features, distributed across 17 so-called feature sets"
    
    #features = [ela_distr", "ela_level", "ela_meta", "basic", "disp", "limo", "nbc", "pca",, "ic"]



    features = ["cm_angle", "cm_conv", "cm_grad", "ela_distr", "ela_level", "ela_meta", 
                "basic", "disp", "limo", "nbc", "pca", "ic"]
    data = {}
    for feature in features:
        try:
            print(feature)
            featureSet = calculate_feature_set(featureObj, feature)
            keys = featureSet.keys()

            for key in keys:
                data[key] = featureSet[key]
        except:
            pass

    print("Es Fitness computation")
    fitness = ESFitness(fitnesses=np.array([EvolutionaryOptimizer.fitness_over_time]), target=target)
    data['FCE'] = fitness.FCE
    data['ERT'] = fitness.ERT
    
    if PerformanceLogger is None:
        Logger = pd.DataFrame(data, index=[0])
    else:
        Logger = PerformanceLogger.append(data,ignore_index=True)
    
    return Logger
    
def computeERT(EvolutionaryOptimizer, target, directory=None):
    #Initial Run
    generation = runOneGeneration(EvolutionaryOptimizer)
    performance = performanceLogger(generation['FeatureObj'], generation['EvolutionaryOptimizer'], target=target)

    #Loop Run--
    while True:
        generation = runOneGeneration(generation['EvolutionaryOptimizer'], generation['Logger'])
        performance = performanceLogger(generation['FeatureObj'], generation['EvolutionaryOptimizer'], target=target, PerformanceLogger = performance)
        print("Budget Used is " + str(generation['EvolutionaryOptimizer'].used_budget))

        if (directory is not None):
            performance.to_csv(directory)
        if (performance['ERT'].iloc[-1] is not None) or (generation['EvolutionaryOptimizer'].used_budget == generation['EvolutionaryOptimizer'].total_budget):
            break

    return generation['EvolutionaryOptimizer'], performance

