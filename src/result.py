import numpy as np
import pandas as pd


class Result():
	def __init__(self):
		self.consolidatedPerformance = pd.DataFrame()
		self.processedPerformance = pd.DataFrame()
		self.calculatedPerformance = pd.DataFrame()
		self.elaFeatures = pd.DataFrame()
		self.processedFeatures = pd.DataFrame()
		self.trainingData = pd.DataFrame()
		self.SBS = ''
		self.excludedFeatures = ['fce_x', 'ert', 'ert-1', 'ert-2', 'ert-3', 'ert-4', 'ert-5', 'ert-6', 'ert-7','ert-8', 'opt', 'ertMax', 'relERT', 'relFCE', 'vbs','sbs', 'ela_meta.quad_simple.cond', 'ic.costs_fun_evals', 'limo.avg.length','limo.avg.length.scaled','limo.avg_length.norm','limo.cor','limo.cor.norm','limo.cor.reg','limo.cor.scaled','limo.costs_fun_evals','limo.length.sd','limo.ratio.sd','limo.sd.max_min_ratio','limo.sd.max_min_ratio.scaled','limo.sd.mean','limo.sd.mean.scaled','limo.sd_mean.norm','limo.sd_mean.reg','limo.sd_ratio.norm','limo.sd_ratio.reg','nbc.costs_fun_evals','pca.costs_fun_evals','Unnamed: 0','ic.eps.ratio']
		self.processedPerf = False
		self.processedSolvers = False
		self.processedELA = False
	
	def addPerformance(self, *args):
		dataframes = []
		for arg in args:
			dataframes.append(arg)
		
		dataframes.append(self.consolidatedPerformance)
		self.consolidatedPerformance = pd.concat(dataframes)

	def addELA(self, *args):
		dataframes = []
		for arg in args:
			dataframes.append(arg)
		
		dataframes.append(self.elaFeatures)
		self.elaFeatures = pd.concat([self.elaFeatures, *args])


	def _elaPrecalculate(self, size):
		#Non destructive preprocessing
		self.processedFeatures = self.elaFeatures

		self.processedFeatures['dim'] = self.processedFeatures['name'].apply(lambda x: x.split("_")[-1])
		self.processedFeatures = self.processedFeatures[self.processedFeatures['dim']==size].drop(columns=['dim'])
		self.processedFeatures['name'] = self.processedFeatures.apply(lambda x: '_'.join(x['name'].split('_')[:-3]), axis=1)
		self.processedELA = True

	def _preCalculation(self):
		#Non destructive preprocessing
		self.processedPerformance = self.consolidatedPerformance
		#split the name identifier to extract the function, instance, algorithm used and dimensions
		self.processedPerformance['function'] = self.processedPerformance['name'].apply(lambda x: x[x.find('_F')+2:x.find('_F')+4].replace('_',''))
		self.processedPerformance['instance'] = self.processedPerformance['name'].apply(lambda x: int(x[x.find('_I')+2:x.find('_I')+4].replace('_','')))
		self.processedPerformance['dimension'] = self.processedPerformance['name'].apply(lambda x: int(x[x.find('_D')+2:x.find('_D')+4].replace('_','')))
		self.processedPerformance['algo'] = self.processedPerformance['name'].apply(lambda x: x[x.find('_Local')+1:x.find('_T')].replace('_',''))
		self.processedPerformance['trial'] =  self.processedPerformance['name'].apply(lambda x: x[x.find('_T')+2:x.find('_T')+4].replace('_',''))

		#Calculate necessary numbers in preparation for calcualte the true performance
		self.consolidatedPerformance['ertMax'] = 10000*self.processedPerformance['dimension']
		self.consolidatedPerformance['relERT'] = self.processedPerformance['ert-8']/self.processedPerformance['ertMax']
		maxFCE = self.processedPerformance.groupby(['function', 'instance', 'dimension','trial'])['fce'].max().reset_index()
		self.processedPerformance = self.processedPerformance.merge(maxFCE, on=['function', 'instance', 'dimension'], how='left', suffixes=('', 'max'))
		self.processedPerformance['fce'] =  self.processedPerformance['fce']+ 1e-8 #adjust FCE to factor in accuracy set when experiment was setup
		self.processedPerformance['relFCE'] = self.processedPerformance.apply(lambda x: 1 + ((np.log10(float(x['fce'])/1e-8))/(np.log10(float(x['fcemax'])/1e-8))), axis=1)

		#Calculate performance based on Rajn's performance measure
		self.processedPerformance['performance'] = self.processedPerformance.apply(lambda x: x['relERT'] if x['relERT'] >0 else x['relFCE'], axis=1)
		self.processedPerf = True

	def _calculateBestSolvers(self):
		#calcluate the virtual best solver
		vbs = self.processedPerformance.groupby(['function', 'dimension', 'instance','trial'])['performance'].min().reset_index()
		vbs=vbs.rename(columns={'performance':'vbs'})
		self.processedPerformance = self.processedPerformance.merge(vbs, on=['function', 'instance', 'dimension','trial'], how='left',  suffixes=('', ''))

		#calculate the single best solver
		sbs =  self.processedPerformance.groupby(['algo'])['performance'].mean().reset_index()
		sbsAlgo = sbs.min()['algo']
		sbs =  self.processedPerformance[ self.processedPerformance['algo']==sbsAlgo][['function', 'dimension', 'instance', 'trial', 'performance']]
		sbs= sbs.rename(columns={'performance':'sbs'})
		self.processedPerformance =  self.processedPerformance.merge(sbs, on=['function', 'instance', 'dimension','trial'], how='left',  suffixes=('', ''))
		self.processedPerformance['VBS-SBS-Gap'] = self.processedPerformance['sbs'] - self.processedPerformance['vbs']
		self.processedSolvers = True


	def calculatePerformance(self):
		self._preCalculation()
		self._calculateBestSolvers()
		self.calculatedPerformance =  self.processedPerformance.groupby(['function', 'dimension', 'algo']).agg({'performance':['mean', 'min', 'max', 'std'], 'vbs':['mean'], 'sbs':['mean']}).reset_index()

		return self.calculatedPerformance

	def createTrainSet(self, size, algorithm=None):
		if not self.processedPerf and not self.processedSolvers:
			self._preCalculation()
			self._calculateBestSolvers()
		elif not self.processedPerf:
			self._preCalculation()
		elif self.processedSolvers:
			self._calculateBestSolvers()

		if not self.processedELA:
			self._elaPrecalculate(size)
	
		self.trainingData = self.processedPerformance.merge(self.elaFeatures, on=['name'], how='left',  suffixes=('', ''))
		self.trainingData['class'] = self.trainingData['performance']==self.trainingData['vbs']

		for feature in self.excludedFeatures:
			try:
				self.trainingData.drop(columns=[feature])
			except:
				pass

		for i in self.elaFeatures.columns:
			try:
				self.trainingData = self.trainingData[self.trainingData[columns[i]].notnull()]

			except:
				pass

		if (algorithm==None):
			training = self.trainingData
		else:
			training = self.trainingData[(self.trainingData['algo']==algorithm)]

		Xtrain = training.iloc[:,12:-1].values
		ytrain = training.iloc[:,-1].values

		return Xtrain, ytrain

