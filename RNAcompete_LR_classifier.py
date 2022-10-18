import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skopt.space import Real
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.model_selection import cross_val_score 

# Read in training data
trainfile = "training_set.txt"
trainset = np.loadtxt(trainfile, delimiter='\t', skiprows=1)

X = trainset[:, 1:len(trainset[0])]
Y = trainset[:, 0]

# Scale data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Read in feature names
afile = open(trainfile, 'r')
featurenames = afile.readline().strip().split('\t')[1:]
afile.close()

# Read in and scale testing data
testfile = "testing_set.txt"
testset = np.loadtxt(testfile, delimiter='\t', skiprows=1)
Xtest = testset[:, 1:len(testset[0])]
Ytest = testset[:, 0]
Xtest = scaler.transform(Xtest)

# Read in and scale ucRBP experiment data
ucrbpfile = "ucrbp_experiment_features.txt"
Xucrbp = np.loadtxt(ucrbpfile, delimiter='\t', skiprows=1)
Xucrbp = scaler.transform(Xucrbp)

# PREPARE TO RUN LOGISTIC REGRESSION
lr = LogisticRegression(solver='liblinear')

# DEFINE PARAMETER GRID
param_grid = {
    'penalty': ['l1'],
    'solver': ['liblinear'],
    'C': Real(low=1e-6, high=100, prior='log-uniform'),
}

# SET UP OPTIMIZER
opt = BayesSearchCV(
    lr,
    param_grid,
    n_iter=30,
    random_state=1234,
    verbose=0,
    n_jobs = 1
)

opt.fit(X, Y)

# FIT PARAMETERS
lr = LogisticRegression(**opt.best_params_)
lr.fit(X, Y)    

# Get probability estimates and AUROC for test set
test_predictions = lr.predict_proba(Xtest)[:,1]
test_auroc = roc_auc_score(Ytest, test_predictions)

# Write probability estimates and AUROC for test set
outfile = open('test_set_probability_estimates.txt', 'w')
outfile.write('probability.estimate\n')
for i in range(len(test_predictions)):
    outfile.write(str(Ytest[i])+'\t'+str(test_predictions[i])+'\n')
outfile.close()

outfile = open('test_set_AUROC.txt', 'w')
outfile.write(str(test_auroc)+"\n")
outfile.close()

# Get probability estimates for ucrbp experiments
ucrbp_predictions = lr.predict_proba(Xucrbp)[:,1]

# Write probability estimates for ucrbp experiments
outfile = open('ucrbp_probability_estimates.txt', 'w')
outfile.write('probability.estimate\n')
for i in range(len(ucrbp_predictions)):
    outfile.write(str(ucrbp_predictions[i])+'\n')
outfile.close()

# Write model coefficients to file
outfile = open("LR_coefficients.txt", 'w')
for i in range(len(lr.coef_[0])):
    outfile.write(featurenames[i]+'\t'+str(lr.coef_[0][i])+'\n')
outfile.close()

filehandler = open("LR_model.sav","wb")
pickle.dump(lr,filehandler)
