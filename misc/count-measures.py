from pypair.biserial import Biserial
from pypair.contingency import BinaryTable, CategoricalTable, ConfusionMatrix
from pypair.continuous import Concordance, CorrelationRatio, Continuous

measures = BinaryTable.measures() + CategoricalTable.measures() + Biserial.measures() + Concordance.measures() + CorrelationRatio.measures() + Continuous.measures() + ConfusionMatrix.measures()
print(measures)
print(len(measures))
