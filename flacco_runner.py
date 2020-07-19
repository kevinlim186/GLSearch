from src.problem import Problem 
from src.logger import Performance
import os
import pandas as pd 

function = [5,6]
fileExtension = 'Function_4_5'
performance = Performance()

baseDIR = "./temp/"
files = os.listdir(baseDIR)
files = [val for val in files if val.endswith("pflacco.csv")]

for i in range(len(files)):
	func= int(files[i].split("_")[1].replace('F',''))
	dim= int(files[i].split("_")[6].replace('D',''))
	filename = files[i].replace('_pflacco.csv', '')

	if func in function :
		problem = Problem(1, func, [1], dim, [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], 1, performance,True)
		problem.currentResults = pd.read_csv(baseDIR+files[i])
		problem.calculateELA()
		problem.saveElaFeat(filename)

performance.saveToCSVELA(fileExtension)