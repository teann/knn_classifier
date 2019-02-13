#!/usr/bin/python3.6 knn_classifier.py

import math
import numpy as np
import json
import pandas as pd
import sys
from pprint import pprint

def main():
	k = int(sys.argv[1])
	train_file = open(sys.argv[2])
	test_file = open(sys.argv[3])

	#sys.stdout = open('out.txt', 'w')


	train_set = json.load(train_file)
	test_set = json.load(test_file)
	train_df = pd.DataFrame(train_set['data'])
	test_df = pd.DataFrame(test_set['data'])

	metadata_df = pd.DataFrame(train_set['metadata'])
	labels = metadata_df['features']
	labels1 = labels[len(labels) - 1]
	#print(labels1[1])
	dictOfLabels = {}
	for la in labels1[1]:
		dictOfLabels[la] = 0

	train_mean = train_df.loc[:,train_df.columns != len(train_df.columns)-1].select_dtypes(include=[np.number]).mean()
	train_std =  train_df.loc[:,train_df.columns != len(train_df.columns)-1].select_dtypes(include=[np.number]).std()
	test_mean = test_df.loc[:,test_df.columns != len(test_df.columns)-1].select_dtypes(include=[np.number]).mean()
	test_std = test_df.loc[:,test_df.columns != len(test_df.columns)-1].select_dtypes(include=[np.number]).std()

	stan_train = getStan(train_mean, train_std, train_df)
	stan_test = getStan(test_mean, test_std, test_df)

	np_tr = np.array(stan_train)
	np_test = np.array(stan_test)
   
	words_train = train_df.select_dtypes(exclude=[np.number])
	words_test = test_df.select_dtypes(exclude=[np.number])

	word_tr = np.array(words_train)
	word_test = np.array(words_test)
	#dist_dict = computeManhattan(np_tr, np_test)
	ham_dict = computeHamming(word_tr, word_test)
	
	#getMin(k, dist_dict, dictOfLabels)
	getMin(k, ham_dict, dictOfLabels)
def getMin(k, distance_dict, labels):
	for i in distance_dict:
		sortedguy = sorted(distance_dict[i], key=lambda tup:tup[1])
		firstk = sortedguy[0:k]
		d = labels.fromkeys(labels, 0)
		for tuples in firstk:
			if tuples[2] in d:
				d[tuples[2]] += 1
		maximum = max(d, key=d.get)
		print(d.values(), maximum)

def getStan(dfmean, dfstd, df):
	num_df = df.loc[:, df.columns != len(df.columns)-1].select_dtypes(include=[np.number])	
	stan_df = (num_df.loc[:,num_df.columns != len(num_df.columns)-1] - dfmean)/dfstd
	stan_df[len(stan_df.columns)] = df.iloc[:,-1]
	return stan_df

def computeManhattan(stan_train, stan_test):
	dict_dist = {}
	instance1 = 0
	for row in stan_test:
		instance1 += 1
	 	list_of_instances = []
	 	dict_dist[instance1] = list_of_instances
	 	instance = 0
	 	rowtosub = row[:len(row) - 1]
	 	for row2 in stan_train:
	 		instance += 1
	 		label = row2[-1]
	 		iterrow = row2[:len(row2) - 1]
	 		newone = abs(iterrow - rowtosub)
	 		distance = np.nansum(newone)
 			dict_dist[instance1].append((instance, distance, label))	
	return dict_dist

def computeHamming(words_train, words_test):
	ham_dict = {}
	instance1 = 0
	for row in words_test:
	#	print(row)
		instance1 += 1
		list_of_instances = []
		ham_dict[instance1] = list_of_instances
		instance = 0
		rowtosub = row[:len(row) - 1]
		for row2 in words_train:
			instance += 1
			label = row2[-1]
			iterrow = row2[:len(row2) - 1]
			newArray = np.equal(iterrow, rowtosub)
			distance = np.size(newArray) - np.count_nonzero(newArray)
			instance_append = ((instance, distance, label))
			ham_dict[instance1].append(instance_append)

	return ham_dict

if __name__ == '__main__':
	main() 