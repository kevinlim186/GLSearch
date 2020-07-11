from src.suites import Suites
from src.logger import Performance

for i in range(1,25):
	performance = Performance()
	suite = Suites(instances=[1,2,3,4,5], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], function=i, performance=performance)

	suite.runTest()
	performance.saveToCSV('Overall_Performance')
