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

	dictOfLabels = collections.OrderedDict()
	for la in labels1[1]:
		dictOfLabels[la] = 0

	oneLabel = labels1[1][0]
	
	metadata_numpy = np.array(train_set['metadata']['features'])
	train_numpy = np.array(train_set['data'])
	train_numpy = np.transpose(train_numpy)
	test_numpy = np.array(test_set['data'])
	test_numpy = np.transpose(test_numpy)
	number_numpy_test = []
	number_numpy_train = []
	word_numpy_test = []
	word_numpy_train = []
	index = 0

	for col_feat in metadata_numpy: 
		if col_feat[1] == 'numeric':
			number_numpy_train.append(train_numpy[index])
			number_numpy_test.append(test_numpy[index])
		elif col_feat[1] != 'numeric' and col_feat[0] != 'label':
			word_numpy_train.append(train_numpy[index])
			word_numpy_test.append(test_numpy[index])
		if (col_feat[0] == 'label'):
			number_numpy_train.append(train_numpy[index])
			number_numpy_test.append(test_numpy[index])
			word_numpy_train.append(train_numpy[index])
			word_numpy_test.append( test_numpy[index])
		index += 1 
	train_df_num = pd.DataFrame(np.transpose(number_numpy_train))
	test_df_num = pd.DataFrame(np.transpose(number_numpy_test))
	train_df_word = pd.DataFrame(np.transpose(word_numpy_train))
	test_df_word = pd.DataFrame(np.transpose(word_numpy_test))

	train_mean = train_df_num.loc[:,train_df_num.columns != len(train_df_num.columns)-1].mean()
	train_std =  train_df_num.loc[:,train_df_num.columns != len(train_df_num.columns)-1].std(ddof = 0)
	train_std = train_std.replace(0, 1)
	num_train = train_df_num
	num_test = test_df_num


	words_train = train_df_word
	words_test = test_df_word


	dist_dict = collections.OrderedDict()
	validator = np.array(test_df.iloc[:,-1])

	if (len(num_test.columns) > 1):
		stan_train = getStan(train_mean, train_std, num_train)
		stan_test = getStan(train_mean, train_std, num_test)

		np_tr = np.array(stan_train)
		np_test = np.array(stan_test)
		dist_dict = computeManhattan(np_tr, np_test)
		confidenceLabels = getMin(k, dist_dict, dictOfLabels, oneLabel, validator)

	if (len(words_test.columns) > 1):
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
		d = labels.fromkeys(labels, 0)
		for tuples in firstk:
			d[tuples[2]] += 1
		maximum = max(d, key=d.get)
		prob_numerator = 0
		prob_denominator = 0
		for tuples in firstk:
			yn = 0
			#calculate confidence for each instance
			weightForThisInstance = 1/((tuples[1])**2 + eps)
			print(weightForThisInstance)

			if (tuples[2] == oneLabel):
				yn = 1
			prob_numerator += weightForThisInstance*yn
			prob_denominator += weightForThisInstance
		confidenceArray.append((i, maximum, prob_numerator/prob_denominator, validator[i - 1]))
	return confidenceArray

def getStan(dfmean, dfstd, df):
	stan_df = (df- dfmean)/dfstd
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
	return dict_dist

def computeHamming(words_train, words_test, ham_dict):
	instance1 = 0
	for row in words_test:
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
#roc_curve algo outlined in slides
def roc_curve(conf_tuples, validator, oneLabel):
	unique, counts = np.unique(validator, return_counts = True)
	count_array = dict(zip(unique, counts))
	conf_tuples = sorted(conf_tuples, key= lambda tup: tup[2], reverse = True)
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