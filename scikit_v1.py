# Imports
from sklearn import svm
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np


#Useful load function
loadData = lambda txt: np.genfromtxt(open(txt,'r'), delimiter = ',')

#Data class created for easier manipulation of the training, labels and test data
class Data(object):

	def __init__(self,features,labels,test):
		self.features = features
		self.labels = labels
		self.test = test

	def do_grid_search(self,C_range,g_range,score,k):
		# Split the dataset in a devlopment and evaluation set
		tuned_parameters = [{'kernel': ['rbf'], 'gamma': g_range,
		            	        'C': C_range}]
		

		cvk = StratifiedKFold(self.labels,k=3)
		print cvk
		print("# Tuning hyper-parameters for %s" % score)

		#Conducts a gridsearch over the defined parameters and selects the parameters that have
		#the best score based on the scoring paramter
		self.classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv = cvk, scoring = score)
		self.classifier.fit(self.features, self.labels)

		print "Best parameters set found on development set:"
		print self.classifier.best_estimator_
	    
	    #Prints the grid scores, as this is stored in the classifier as a variable
		print "Grid scores on development set:"
		for params, mean_score, scores in self.classifier.grid_scores_:
			print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

	#def semisupervised_boost():

	def pca_transform(self,n_components,whiten_bool):

		pca = PCA(n_components,whiten_bool)

		self.features = pca.fit_transform(self.features)
		self.test = pca.transform(self.test)

	def estimate_score(self,k):

		scores = cv.cross_val_score(self.classifier.best_estimator_,self.features,self.labels,cv = k)
		print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))



#Main script
if __name__ == '__main__':

	#Loading csv files
	trainFeatures = loadData('train.csv')
	trainLabels = loadData('trainLabels.csv')
	testFeatures = loadData('test.csv')

	#Create nstance of Data class
	data = Data(trainFeatures,trainLabels, testFeatures)

	#Estimator parameter grid search
	C_range = [1, 100, 1000, 10000, 100000, 1000000]
	g_range = [0.0001,0.001,0.01, 0.1, 0.2,0.27777778,0.3]
	#Score for parameter evaluation, also possible: f1, precision, accuracy, recall...
	score = 'accuracy'
	#K folds for cross validation of score evaluation
	k = 5

	#Methods and data manipulation
	data.pca_transform(n_components = 12, whiten_bool= True)
	data.do_grid_search(C_range,g_range,score,k)
	data.estimate_score(k=60)

	# Write results file in the form specified under the competition details
	resultsFile = open('results.csv','w')
	#best classifier obtained from grid search is used
	results = data.classifier.best_estimator_.predict(data.test)
	count = 0
	resultsFile.write('Id,Solution\n')

	for label in results:
		count += 1
		resultsFile.write('%d,%d\n' % (count,label))
		
	#solution = classifier.predict(testFeatures)


	resultsFile.close()










