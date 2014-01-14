# Imports
from sklearn import svm
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

classifier = svm.SVC(gamma = 0.001, C=100.)
classifier.fit(trainingFeatures, trainingLabels)

#count = 0
#for ii in range(1,101):
	
#	print str(classifier.predict(trainingFeatures[-ii])) + '' + str(trainingLabels[-ii])
#	if classifier.predict(trainingFeatures[-ii])[0] == trainingLabels[-ii]:
#		#print 'match'
#		count +=1
		
#print float(count)/100.

# Write result

resultsFile = open('results.csv','a+')
resultsFileWriter = csv.writer(resultsFile)


id = 0
#resultsFileWriter.writerow(['Id','Solution'])
#for sample in testFeatures:
#	id += 1
#	solution = classifier.predict(testFeatures[id-1])[0]
#	resultsFileWriter.writerow([str(id),str(solution)])
	#resultsFileWriter.writerow([str(solution)])
	
solution = classifier.predict(testFeatures)


resultsFile.close()
	


