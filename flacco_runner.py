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
	func= int(files[i].split("_")[1].replace('F',''))
	dim= int(files[i].split("_")[-6].replace('D',''))
	filename = files[i].replace('_pflacco.csv', '')
	_printProgressBar(i, len(files))
	if func in function :
		problem = Problem(1, func, [1], dim, [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], 1, performance,True)
		problem.currentResults = pd.read_csv(baseDIR+files[i])
		problem.calculateELA()
		problem.saveElaFeat(filename)

performance.saveToCSVELA(fileExtension)