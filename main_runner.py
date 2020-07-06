from src.suites import Suites
from src.logger import Performance

performance = Performance(True)
suite = Suites(instances=[1,2,3,4,5], baseBudget=10000, dimensions=[2,3,5,10,20], esconfig=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], function=1, performance=performance)

suite.runTest()
