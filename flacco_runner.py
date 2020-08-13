from src.problem import Problem 
from src.logger import Performance
import os
import pandas as pd 

#choose for parallel run
function = [1,2,3,4,5]
#function = [10,11,12,13,14]
#function = [15,16,18,19]
#function = [20,22,23,24]

fileExtension = 'Function'
for i in function:
	fileExtension = fileExtension + "_" + str(i)


performance = Performance()

baseDIR = "./temp/"
files = os.listdir(baseDIR)
files = [val for val in files if val.endswith(".csv")]
errorlog = pd.DataFrame()

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
	func= int(files[i].split("_")[1].replace('F',''))
	if func in function :
		dim= int(files[i].split("_")[-5].replace('D',''))
		endRef = int(files[i].split("_")[-1].replace('B','').replace('.csv', ''))
		

		#limit the number of features to 50D, 100D, 200D 
		for j in [5,50,100,200]:
#			sample = j * dim
			_lambda = 4+floor(3*log(dim))
			sample = _lambda * j
			begRef = endRef - sample
			filename = files[i].replace('.csv', '_ela_sample_populationBased' + str(j))

			problem = Problem(1, func, [1], dim, [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], 1, performance,True)
			historicalPath = pd.read_csv(baseDIR+files[i]).iloc[begRef:endRef,]
			problem.currentResults = historicalPath
			try:
				problem.calculateELA()
				problem.saveElaFeat(filename)
				performance.saveToCSVELA(fileExtension)
			except:
				errorlog.append({'name':filename}, ignore_index = True)
				errorlog.to_csv('./perf/error_log.csv',index=False)