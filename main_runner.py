from src.suites import Suites
from src.logger import Performance

performance = Performance()
#function 6 to 7
for i in range(6,8):
	suite = Suites(instances=[1,2,3,4,5], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], function=i, performance=performance, pflacco=False)

	suite.runTest()


