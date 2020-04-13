from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
import bbobbenchmarks.bbobbenchmarks as bn
from modea import Algorithms, Parameters, Individual, Mutation, Utils, Recombination, Selection, Sampling
from modea.Utils import getOpts, getVals, reprToString, options, initializable_parameters
import os
import numpy as np
from functools import partial
import math

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
        function['function'] #returns a function 
        function['function']([1,2,3]) #evaluates the function based on the given parameters
    
    Functions created here comes from http://coco.lri.fr/backup/COCOdoc/index.html# adapted to work in Python 3.
    '''

    functionNamesNoiseless = [ 
                    'Noise-free Sphere function', 
                    'Separable ellipsoid with monotone transformation', 
                    'Rastrigin with monotone transformation separable "condition" 10', 
                    'skew Rastrigin-Bueche, condition 10, skew-"condition" 100', 
                    'Linear slope', 
                    'Attractive sector function',
                    'Step-ellipsoid, condition 100, noise-free',
                    'Rosenbrock noise-free',
                    'Rosenbrock, rotated', 
                    'Ellipsoid with monotone transformation, condition 1e6', 
                    'Discus (tablet) with monotone transformation, condition 1e6', 
                    'Bent cigar with asymmetric space distortion, condition 1e6',
                    'Sharp ridge',
                    'Sum of different powers, between x^2 and x^6, noise-free',
                    'Rastrigin with asymmetric non-linear distortion, "condition" 10',
                    'Weierstrass, condition 100',
                    'Schaffers F7 with asymmetric non-linear transformation, condition 10',
                    'Schaffers F7 with asymmetric non-linear transformation, condition 1000',
                    'F8F2 sum of Griewank-Rosenbrock 2-D blocks, noise-free',
                    'Schwefel with tridiagonal variable transformation',
                    'Gallagher with 101 Gaussian peaks, condition up to 1000, one global rotation, noise-free',
                    'Gallagher with 21 Gaussian peaks, condition up to 1000, one global rotation',
                    'Katsuura function',
                    'Lunacek bi-Rastrigin, condition 100',
                    'The number of legs the animal (default is 4)'
                    ]
    
    functionNamesNoisy = [
                    'Sphere with moderate Gaussian noise',
                    'Sphere with moderate uniform noise',
                    'Sphere with moderate seldom Cauchy noise', 
                    'Rosenbrock with moderate Gaussian noise', 
                    'Rosenbrock with moderate uniform noise',
                    'Rosenbrock with moderate seldom Cauchy noise',
                    'Sphere with Gaussian noise',
                    'Sphere with uniform noise',
                    'Sphere with seldom Cauchy noise',
                    'Rosenbrock with Gaussian noise',
                    'Rosenbrock with uniform noise',
                    'Rosenbrock with seldom Cauchy noise',
                    'Step ellipsoid with Gaussian noise',
                    'Step ellipsoid with uniform noise',
                    'Step ellipsoid with seldom Cauchy noise',
                    'Ellipsoid with Gaussian noise',
                    'Ellipsoid with uniform noise',
                    'Ellipsoid with seldom Cauchy noise',
                    'Different Powers with Gaussian noise',
                    'Different Powers with uniform noise',
                    'Different Powers with seldom Cauchy noise',
                    'Schaffer\'s F7 with Gaussian noise',
                    'Schaffer\'s F7 with uniform noise',
                    'Schaffer\'s F7 with seldom Cauchy noise',
                    'Composite Griewank-Rosenbrock with Gaussian noise',
                    'Composite Griewank-Rosenbrock with uniform noise',
                    'Composite Griewank-Rosenbrock with seldom Cauchy noise',
                    'Gallagher\'s Gaussian Peaks 101-me with Gaussian noise',
                    'Gallagher\'s Gaussian Peaks 101-me with uniform noise',
                    'Gallagher\'s Gaussian Peaks 101-me with seldom Cauchy noise'
            ]
    if  isinstance(instanceNumber, int):
        return print('Please enter a valid instance number')

    if functionID >=0 and functionID <=24:
        functionIndex = functionID - 1
        functionName = functionNamesNoiseless[functionIndex]
        functionAttr = 'F' + str(functionID)
        function = getattr(bn, functionAttr)(instanceNumber)
        returnValue = {}
        returnValue['name'] = functionName
        returnValue['function'] = function
        return returnValue
    elif functionID >=101 and functionID <=130:
        functionIndex = functionID - 101
        functionName = functionNamesNoisy[functionIndex]
        functionAttr = 'F' + str(functionID)
        function = getattr(bn, functionAttr)(instanceNumber)
        returnValue = {}
        returnValue['name'] = functionName
        returnValue['function'] = function
        return returnValue
    else:
        print("Please choose between Function 1 to 24 for noiseless functions and Function 101 to 130 for noisy function")
