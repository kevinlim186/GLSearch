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
		self.classificationCost = pd.DataFrame()
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
		self.elaFeatures = pd.concat(dataframes)


	def _elaPrecalculate(self, size):
		#Non destructive preprocessing
		self.processedFeatures = self.elaFeatures

		self.processedFeatures['dim'] = self.processedFeatures['name'].apply(lambda x: x.split("_")[-1])
		self.processedFeatures = self.processedFeatures[self.processedFeatures['dim']==size].drop(columns=['dim'])
		self.processedFeatures['name'] = self.processedFeatures['name'].apply(lambda x: '_'.join(x.split('_')[:-3]))
		
		#split the name identifier to extract the function, instance, algorithm used and dimensions
		self.processedFeatures['function'] =  self.processedFeatures['name'].str.extract('(_F[0-9]+)')
		self.processedFeatures['function'] = self.processedFeatures['function'].apply(lambda x: x.replace('_F',''))

		self.processedFeatures['instance'] =  self.processedFeatures['name'].str.extract('(_I[0-9]+)')
		self.processedFeatures['instance'] = self.processedFeatures['instance'].apply(lambda x: x.replace('_I',''))

		self.processedFeatures['dimension'] =  self.processedFeatures['name'].str.extract('(_D[0-9]+)')
		self.processedFeatures['dimension'] = self.processedFeatures['dimension'].apply(lambda x: x.replace('_D',''))



		self.processedFeatures['trial'] =  self.processedFeatures['name'].str.extract('(_T[0-9]+)')
		self.processedFeatures['trial'] = self.processedFeatures['trial'].apply(lambda x: x.replace('_T',''))

		self.processedFeatures['budget'] = self.processedFeatures['name'].str.extract('(_B[0-9]+)')
		self.processedFeatures['budget'] = self.processedFeatures['budget'].apply(lambda x: x.replace('_B',''))

		self.processedFeatures['algo'] = self.processedFeatures['name'].apply(lambda x: x[x.find('_Local')+1:x.find('_T')].replace('_',''))

		self.processedFeatures[['function','instance', 'dimension', 'trial', 'budget']]= self.processedFeatures[['function','instance', 'dimension', 'trial', 'budget']].astype('int64')

		self.processedELA = True

	def _preCalculation(self):
		#Non destructive preprocessing
		self.processedPerformance = self.consolidatedPerformance
		#split the name identifier to extract the function, instance, algorithm used and dimensions
		self.processedPerformance['function'] =  self.processedPerformance['name'].str.extract('(_F[0-9]+)')
		self.processedPerformance['function'] = self.processedPerformance['function'].apply(lambda x: x.replace('_F',''))

		self.processedPerformance['instance'] =  self.processedPerformance['name'].str.extract('(_I[0-9]+)')
		self.processedPerformance['instance'] = self.processedPerformance['instance'].apply(lambda x: x.replace('_I',''))

		self.processedPerformance['dimension'] =  self.processedPerformance['name'].str.extract('(_D[0-9]+)')
		self.processedPerformance['dimension'] = self.processedPerformance['dimension'].apply(lambda x: x.replace('_D',''))



		self.processedPerformance['trial'] =  self.processedPerformance['name'].str.extract('(_T[0-9]+)')
		self.processedPerformance['trial'] = self.processedPerformance['trial'].apply(lambda x: x.replace('_T',''))

		self.processedPerformance['budget'] = self.processedPerformance['name'].str.extract('(_B[0-9]+)')
		self.processedPerformance['budget'] = self.processedPerformance['budget'].apply(lambda x: x.replace('_B',''))

		self.processedPerformance['algo'] = self.processedPerformance['name'].apply(lambda x: x[x.find('_Local')+1:x.find('_T')].replace('_',''))

		self.processedPerformance[['function','instance', 'dimension', 'trial', 'budget']]= self.processedPerformance[['function','instance', 'dimension', 'trial', 'budget']].astype('int64')
		
		#remove first generation sample since ELA could not be computed
		self.processedPerformance = self.processedPerformance[self.processedPerformance['budget'] >100]

		#Calculate necessary numbers in preparation for calculate the true performance
		self.processedPerformance['ertMax'] = 10000*self.processedPerformance['dimension']
		self.processedPerformance['relERT'] = self.processedPerformance['ert-8']/self.processedPerformance['ertMax']
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

		self.processedPerformance['cost'] = self.processedPerformance['performance']-self.processedPerformance['vbs']

		self.classificationCost = self.processedPerformance.pivot_table(index=['function', 'dimension','instance', 'budget', 'trial'], columns = 'algo', values='cost').reset_index().sort_values(['function', 'dimension','instance', 'budget'], ascending=True)
		self.classificationCost['Local:Base'] = self.classificationCost['Local:Base'].bfill()
		self.classificationCost['Local:bfgs0.1'] = self.classificationCost['Local:bfgs0.1'].ffill()
		self.classificationCost['Local:bfgs0.3'] = self.classificationCost['Local:bfgs0.3'].ffill()
		self.classificationCost['Local:nedler'] = self.classificationCost['Local:nedler'].ffill()
		self.processedSolvers = True


	def calculatePerformance(self):
		self._preCalculation()
		self._calculateBestSolvers()
		self.calculatedPerformance =  self.processedPerformance.groupby(['function', 'dimension', 'algo']).agg({'performance':['mean', 'min', 'max', 'std'], 'vbs':['mean'], 'sbs':['mean']}).reset_index()

		return self.calculatedPerformance

	def createTrainSet(self, size, algorithm=None, reset=False):
		if reset:
			self._reset()
		if not self.processedPerf and not self.processedSolvers:
			self._preCalculation()
			self._calculateBestSolvers()
		elif not self.processedPerf:
			self._preCalculation()
		elif self.processedSolvers:
			self._calculateBestSolvers()

		if not self.processedELA:
			self._elaPrecalculate(size)
	
		self.trainingData = self.processedFeatures.merge(self.classificationCost, on= ['function','instance','dimension','trial', 'budget'])

		for feature in self.excludedFeatures:
			try:
				self.trainingData.drop(columns=[feature])
			except:
				pass

		for i in self.elaFeatures.columns:
			try:
				self.trainingData = self.trainingData[self.trainingData[i].notnull()]

			except:
				pass
		
		if (algorithm==None):
				training = self.trainingData
		else:
			training = self.trainingData[(self.trainingData['algo']==algorithm)]


		Xtrain = training.iloc[:,12:-4].values
		ycost = training.iloc[:,-4:].values

		return Xtrain, ycost


	def _reset(self):
		self.processedSolvers = False
		self.processedPerf = False
		self.processedELA = False