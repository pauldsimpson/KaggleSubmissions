# Imports
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np

# Opening training set features (X)
trainingFeaturesFile = open('train.csv','r')
trainingFeaturesReader = csv.reader(trainingFeaturesFile)
# Moving to numpy array
trainingFeatures = []
for sample in trainingFeaturesReader:
	newSample = []
	for feature in sample:
		newSample.append(float(feature))
		
	trainingFeatures.append(newSample)

# Conversion to numpy array
trainingFeatures = np.array(trainingFeatures)

# Opening training set labels (y)
trainingLabelsFile = open('trainLabels.csv','r')
trainingLabelsReader = csv.reader(trainingLabelsFile)
# Moving to numpy array
trainingLabels = []
for sample in trainingLabelsReader:
	trainingLabels.append(int(sample[0]))

#Conversion to numpy array
trainingLabels = np.array(trainingLabels)

testFeaturesFile = open('test.csv','r')
testFeaturesReader = csv.reader(testFeaturesFile)
# Moving to numpy array
testFeatures = []
for sample in testFeaturesReader:
	newSample = []
	for feature in sample:
		newSample.append(float(feature))
		
	testFeatures.append(newSample)

# Conversion to numpy array
testFeatures = np.array(testFeatures)

# Test classifier

# # rbf kernel classifier
# classifier = svm.SVC(gamma = 0.001, C=100)
# # linear kernel classifier
# #classifier = svm.SVC(kernel = 'linear', C=1)

# # Train the classifier on the training data
# classifier.fit(trainingFeatures, trainingLabels)

# # K fold CROSS VALIDATION
# k = 5
# crossValidationScores = cross_validation.cross_val_score(classifier,trainingFeatures,trainingLabels, cv=k)

# print scores
# print np.average(crossValidaticores)

#############################################################################################

# Split the dataset in a devlopment and evaluation set
trainingFeaturesDevelopmentSet, trainingFeaturesEvaluationSet, trainingLabelsDevelopmentSet, trainingLabelsEvaluationSet = train_test_split(
    trainingFeatures, trainingLabels, test_size=0.5, random_state=0)

# Grid seach parameters, searching over rbf and linear kernel
C_range = np.arange(1, 100, 1)
#g_range = np.arange(1e-3,1e-2,1e-4)
g_range = [0.01,0.02,0.03,0.04,0.05]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': g_range,
            	        'C': C_range}]

scores = ['accuracy']

#This loop enables us to compare other scoring functions. This competition hoever evaluates
#using accuracy
for score in scores:

	print("# Tuning hyper-parameters for %s" % score)

	#Conducts a gridsearch over the defined parameters and selects the parameters that have
	#the best score based on the scoring paramter
	classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv = 20, scoring = score)
	classifier.fit(trainingFeaturesDevelopmentSet, trainingLabelsDevelopmentSet)

	print("Best parameters set found on development set: \n")
	print(classifier.best_estimator_)
    
    #Prints the grid scores, as this is stored in the classifier as a variable
	print("Grid scores on development set:")
	for params, mean_score, scores in classifier.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))


	print("Detailed classification report:")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	labels_true, labels_pred = trainingLabelsEvaluationSet, classifier.predict(trainingFeaturesEvaluationSet)
	print(classification_report(labels_true, labels_pred))
    




# Write results file in the form specified under the competition details
resultsFile = open('results.csv','w')

count = 0
resultsFile.write('Id,Solution\n')
for sample in testFeatures:
	count += 1
	solution = classifier.predict(testFeatures[count-1])[0]
	resultsFile.write('%d,%d\n' % (count,solution))
	
#solution = classifier.predict(testFeatures)


resultsFile.close()
	


