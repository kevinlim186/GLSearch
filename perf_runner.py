from src.problem import Problem 
from src.logger import Performance
import os
import pandas as pd 
import math

allocation = [0,1]
#allocation = [.1,.2]
#allocation = [.2,.3]
#allocation = [.3,.4]
#allocation = [.4,.5]
#allocation = [.6,.7]
#allocation = [.7,.8]
#allocation = [.9,1]

fileExtension = 'allocation_'+str(allocation[0])+'_'+str(allocation[1])
performance = Performance()

baseDIR = "./temp/"
files = os.listdir(baseDIR)
files = [val for val in files if val.endswith(".csv")]
total = len(files)
files = files[math.ceil(total*allocation[0]):math.floor(total*allocation[1])]
errorlog =pd.DataFrame()

def _printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
		printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
	# Print New Line on Complete
	if iteration == total: 
		print()


for i in range(len(files)):
	_printProgressBar(i, len(files))
	print(files[i])
	func= int(files[i].split("_")[1].replace('F',''))

	dim= int(files[i].split("_")[-5].replace('D',''))
		
	filename = files[i].replace('.csv', '')
 
	problem = Problem(1, func, [1], dim, [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], 1, performance,False)
	historicalPath = pd.read_csv(baseDIR+files[i])
	problem.currentResults = historicalPath
	try:
		problem.calculatePerformance(filename)
		performance.saveToCSVPerformance(fileExtension)
	except:
		errorlog.append({'name':filename}, ignore_index = True)
		errorlog.to_csv('./perf/error_log.csv',index=False)