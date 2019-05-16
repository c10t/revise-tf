import pandas as pd

train = pd.read_csv('exoTrain.csv')
test = pd.read_csv('exoTest.csv')

print('Train:\n', train.head())
print('Test:\n', test.head())

# TFBT estimator assumes label starting with zero
# Features in datasets are 1 and 2
x_train = train.drop('LABEL', axis=1)
y_train = train.LABEL - 1
x_test = test.drop('LABEL', axis=1)
y_test = test.LABEL - 1
