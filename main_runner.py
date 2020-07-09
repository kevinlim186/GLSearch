from src.suites import Suites
from src.logger import Performance

performance = Performance()
suite = Suites(instances=[1], baseBudget=10000, dimensions=[2], esconfig=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1], function=1, performance=performance)

suite.runTest()
performance.saveToCSV()