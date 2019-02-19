#!/usr/bin/python3.6 roc_curve.py

import math
import numpy as np
import json
import pandas as pd
import sys
from pprint import pprint
import collections


def main():
	k = int(sys.argv[1])
	train_file = open(sys.argv[2])
	test_file = open(sys.argv[3])

	train_set = json.load(train_file)
	test_set = json.load(test_file)
	train_df = pd.DataFrame(train_set['data'])
	test_df = pd.DataFrame(test_set['data'])

	metadata_df = pd.DataFrame(train_set['metadata'])
	labels = metadata_df['features']
	labels1 = labels[len(labels) - 1]
	#print(labels1[1])

	#sys.stdout = open("out.txt" , "w")
	dictOfLabels = collections.OrderedDict()
	for la in labels1[1]:
		dictOfLabels[la] = 0

	oneLabel = labels1[1][0]

	train_mean = train_df.loc[:,train_df.columns != len(train_df.columns)-1].select_dtypes(include=[np.number]).mean()
	train_std =  train_df.loc[:,train_df.columns != len(train_df.columns)-1].select_dtypes(include=[np.number]).std(ddof = 0)
	train_std = train_std.replace(0, 1)

	num_train = train_df.select_dtypes(include=[np.number])
	num_test = test_df.select_dtypes(include=[np.number])


	words_train = train_df.select_dtypes(exclude=[np.number])
	words_test = test_df.select_dtypes(exclude=[np.number])

	dist_dict = collections.OrderedDict()
	validator = np.array(test_df.iloc[:,-1])

	if (len(num_test.columns) > 0):
		stan_train = getStan(train_mean, train_std, num_train)
		stan_test = getStan(train_mean, train_std, num_test)

		np_tr = np.array(stan_train)
		np_test = np.array(stan_test)
		dist_dict = computeManhattan(np_tr, np_test)
		confidenceLabels = getMin(k, dist_dict, dictOfLabels, oneLabel, validator)

	if (len(words_test.columns) > 0):
		word_tr = np.array(words_train)
		word_test = np.array(words_test)
		ham_dict = computeHamming(word_tr, word_test, dist_dict)
		confidenceLabels = getMin(k, ham_dict, dictOfLabels, oneLabel, validator)

	roc_curve(confidenceLabels, validator, oneLabel)

def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)


def getMin(k, distance_dict, labels, oneLabel, validator):
	eps = 1*10**-5
	confidenceArray = []
	for i in distance_dict:
		sortedguy = sorted(distance_dict[i], key=lambda tup:tup[1])
		firstk = sortedguy[0:k]
#	print(sortedguy)
	#	print(firstk)
		d = labels.fromkeys(labels, 0)
	#	print(d)
		for tuples in firstk:
			d[tuples[2]] += 1
		maximum = max(d, key=d.get)
		#print(firstk)
		prob_numerator = 0
		prob_denominator = 0
		for tuples in firstk:
			#GET THE WEIGHT
			yn = 0
			weightForThisInstance = 1/((tuples[1])**2 + eps)
			if (tuples[2] == oneLabel):
				yn = 1
			prob_numerator += weightForThisInstance*yn
			prob_denominator += weightForThisInstance
		confidenceArray.append((i, maximum, prob_numerator/prob_denominator, validator[i - 1]))
	return confidenceArray

def getStan(dfmean, dfstd, df):
	stan_df = (df- dfmean)/dfstd
	#stan_df[len(stan_df.columns)] = df.iloc[:,-1]
	stan_df[len(stan_df.columns) - 1] = df.iloc[:,-1]
	return stan_df

def computeManhattan(stan_train, stan_test):
	dict_dist = collections.OrderedDict()
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
			newone = np.absolute(np.subtract(rowtosub, iterrow))
			distance = np.nansum(newone)
			dict_dist[instance1].append((instance, distance, label))	
	#print(dict_dist[607])
	return dict_dist

def computeHamming(words_train, words_test, ham_dict):
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

def roc_curve(conf_tuples, validator, oneLabel):
	unique, counts = np.unique(validator, return_counts = True)
	count_array = dict(zip(unique, counts))
	conf_tuples = sorted(conf_tuples, key= lambda tup: tup[2], reverse = True)
	#print(*conf_tuples, sep='\n')

	#print(conf_tuples)
	num_pos = counts[0]
	num_neg = counts[1]
	TP = 0
	FP = 0
	last_TP = 0

	for i in range(len(conf_tuples)):
		if (i > 0) and (conf_tuples[i][2] != conf_tuples[i - 1][2]) and (conf_tuples[i][3] != oneLabel) and (TP > last_TP):
			FPR = FP/num_neg
			TPR = TP/num_pos
			print(str(FPR) + ',' + str(TPR))
			last_TP = TP
		if (conf_tuples[i][3] == oneLabel):
			TP += 1
		else:
			FP += 1
	FPR = FP/num_neg
	TPR = TP/num_pos
	print(str(FPR) + ',' + str(TPR))





if __name__ == '__main__':
	main() 