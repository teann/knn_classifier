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


	dist_dict = computeManhattan(np_tr, np_test)
	# dist_dict_ham = computeHamming(words_train, words_test)

	print dist_dict

def getStan(dfmean, dfstd, df):
	num_df = df.loc[:, df.columns != len(df.columns)-1].select_dtypes(include=[np.number])	
	stan_df = (num_df.loc[:,num_df.columns != len(num_df.columns)-1] - dfmean)/dfstd
	stan_df[len(stan_df.columns)] = df.iloc[:,-1]
	return stan_df

def computeManhattan(stan_train, stan_test):
#	print(stan_train)
#	print(stan_test)
	dict_dist = {}
	instance1 = 0
	for row in np.nditer(stan_test):
		instance1 += 1
	#	print('here')
		print(instance1)
	#	print(row)
	 	list_of_instances = []
	 	dict_dist[instance1] = list_of_instances
	 	instance = 0
	 	for row2 in np.nditer(stan_train):
	 		instance += 1
	 		distance_for_this_instance = 0
	 		label = row2[-1]
	 		iterrow = row2[:len(row2) - 1]
	 		which_feature = 0
	 		for i in np.nditer(iterrow):
	 			which_feature += 1
	 			distance = abs(i - row[which_feature])
	 			distance_for_this_instance += distance
	 			dict_dist[instance1].append((instance1, distance_for_this_instance, label))
	 		# distance_vector = abs(iterrow - row[:len(row) - 1])
	 		# distance_for_this_instance = distance_vector.sum()
	 		# dict_dist[index].append((index_train, distance_for_this_instance, label)) 
	return dict_dist

# def computeHamming(words_train, words_test):
# 	ham_dict = {}
# 	for index, row in words_test.iterrows():
# 		print index
# 		list_of_instances = []
# 		ham_dict[index] = list_of_instances
# 		for index_train, row_train in words_train.iterrows():
# 			distance_for_this_instance = 0
# 			label = row_train.tail(1)
# 			iterrow = row_train[:len(row_train) - 1]
# 			row2 = row[:len(row) - 1]
# 			distvec = iterrow.where(iterrow.values!=row2.values)
# 			distance = distvec.count()
# 			instance_append = ((index_train, distance, label))
# 			ham_dict[index].append(instance_append)

# 	return ham_dict

if __name__ == '__main__':
	main() 