'''
BPD_preprocess.py: Just contains a preprocessing function that
puts all the data into one table. This function is called by the
classification scripts to load the data.
'''

import numpy as np
import pandas as pd
import os

def preprocessData():
	#--------------#
	# Loading data #
	#--------------#

	'''
	I'm using dicts containing dicts and data to store data in a directory-like format
	to match the way they are actually organized in the filesystem.
	Some data is 1 level deep into the software system folder, some is 2 levels.

	data
		data['eclipse']
			data['eclipse']['biweekly-ck-values']
				data['eclipse']['biweekly-ck-values']['eclipse-class-cbo'] <- data
			...
			data['eclipse']['bug-metrics'] <- data
			...
		...

	Probably not good form but it makes enough sense to me.
	'''

	print('Loading data...')
	data = {}

	sw_systems = os.listdir('../data/')
	data_lv1 = os.listdir('../data/eclipse/') #I use eclipse here but it's the same for all sw_systems
	lv1_dirs = [s for s in data_lv1 if '.csv' not in s] #which names in lv1 are directories
	data_lv1 = [x.replace('.csv', '') for x in data_lv1]

	for sw in sw_systems:
		new_data = {}
		for d1 in data_lv1:
			if d1 not in lv1_dirs:
				new_data[d1] = pd.read_csv('../data/%s/%s' % (sw, d1 + '.csv'))
			else:
				new_data[d1] = {}
				for d2 in os.listdir('../data/%s/%s' % (sw, d1)):
					d2 = d2.replace('.csv', '')
					#Biweekly tables do not have column headers.
					#Also contain the name of the sw system, removing so they're the same for all systems
					if ('biweekly' in d1):
						d2_new = d2.replace(sw+'-', '')
						new_data[d1][d2_new] = pd.read_csv('../data/%s/%s/%s' % (sw, d1, d2 + '.csv'), header=None)
						new_data[d1][d2_new] = new_data[d1][d2_new].rename(columns={0: 'classname '})
					else:
						new_data[d1][d2] = pd.read_csv('../data/%s/%s/%s' % (sw, d1, d2 + '.csv'))
		
		data[sw] = new_data

	#--------------------------------#
	# Create labels and combine data #
	#--------------------------------#

	sw_systems_to_include = ['eclipse',
							 'equinox',
							 'lucene',
							 'mylyn',
							 'pde',
							 ]

	lv1_data_to_include = ['bug-metrics',
						   'change-metrics',
						   'complexity-code-change',
						   'single-version-ck-oo',
						   ]

	#tuples: (directory, data)
	lv2_data_to_include = [('biweekly-ck-values', 'class-cbo'),
						   ('biweekly-ck-values', 'class-dit'),
						   ('biweekly-ck-values', 'class-lcom'),
						   ('biweekly-ck-values', 'class-noc'),
						   ('biweekly-ck-values', 'class-rfc'),
						   ('biweekly-ck-values', 'class-wmc'),
						   ('biweekly-oo-values', 'class-fanIn'),
						   ('biweekly-oo-values', 'class-fanOut'),
						   ('biweekly-oo-values', 'class-numberOfAttributes'),
						   ('biweekly-oo-values', 'class-numberOfAttributesInherited'),
						   ('biweekly-oo-values', 'class-numberOfLinesOfCode'),
						   ('biweekly-oo-values', 'class-numberOfMethods'),
						   ('biweekly-oo-values', 'class-numberOfMethodsInherited'),
						   ('biweekly-oo-values', 'class-numberOfPrivateAttributes'),
						   ('biweekly-oo-values', 'class-numberOfPrivateMethods'),
						   ('biweekly-oo-values', 'class-numberOfPublicAttributes'),
						   ('biweekly-oo-values', 'class-numberOfPublicMethods'),
						   ('churn', 'churn'),
						   ('churn', 'exp-churn'),
						   ('churn', 'lin-churn'),
						   ('churn', 'log-churn'),
						   ('churn', 'weighted-churn'),
						   ('entropy', 'ent'),
						   ('entropy', 'exp-ent'),
						   ('entropy', 'lin-ent'),
						   ('entropy', 'log-ent'),
						   ('entropy', 'weighted-ent'),
						   ]


	def createLabels(bugs, nonTrivialBugs, majorBugs, criticalBugs, highPriorityBugs):
		'''
		Create labels from the 5 bugs columns.
		If all values are 0, label 'none'
		If "bugs" value is nonzero and others are 0, label 'triv'
		If others are nonzero, label 'nontriv'
		'''
		if (bugs + nonTrivialBugs + majorBugs + criticalBugs + highPriorityBugs == 0): #this is assuming there aren't "negative" bug counts
			return 'none'
		elif (bugs != 0) and (nonTrivialBugs + majorBugs + criticalBugs + highPriorityBugs == 0):
			return 'triv'
		else:
			return 'ntrv'

	def combineData(combined_data, ind_data, sw, name):
		'''
		Adds a dataframe (ind_data) to the dataset to be used in classification (combined_data).
		This removes any 'bugs' columns that were used to create labels and any empty ' ' column,
		adds the 'name' parameter to the beginning of each column name (so as to not
		have multiple columns with the same name in the combined set) and appends
		the columns to the combined dataset.
		'''
		#Yes, there is whitespace in the column names for some reason
		#errors='ignore' makes it so a column is dropped IF it exists
		new_ind_data = ind_data.drop(['classname ', ' bugs ', ' nonTrivialBugs ', ' majorBugs ', ' criticalBugs ', ' highPriorityBugs ', ' '], axis=1, errors='ignore')
		new_ind_data.columns = name + '_' + new_ind_data.columns.astype(str)
		
		#Remove "extra" classnames from biweekly data
		#Clarification: there are two sets of classnames - a larger set which appears in the biweekly tables
		#and a smaller set in all the other data. From testing, all classnames in the smaller set are
		#indeed in the larger set. So removing classnames from the larger set which do not appear in
		#the smaller set makes the classname sets match. This is true for all the software systems.
		rows_to_remove = ind_data[~ind_data['classname '].isin(data[sw]['bug-metrics']['classname '])].index
		new_ind_data = new_ind_data.drop(rows_to_remove)
		new_ind_data = new_ind_data.reset_index(drop=True)
		return pd.concat([combined_data, new_ind_data], axis=1)


	#labels and combined data for ALL of the selected software systems
	print('Preprocessing...')
	labels = pd.Series(dtype='object')
	combined_data = pd.DataFrame()

	for sw in sw_systems_to_include:
		#labels and combined data for 1 software system at a time
		labels_1sw = data[sw]['bug-metrics'].apply(lambda row: createLabels(row[' bugs '], row[' nonTrivialBugs '], row[' majorBugs '], row[' criticalBugs '], row[' highPriorityBugs ']), axis=1)
		labels_1sw.index = data[sw]['bug-metrics']['classname ']
		
		combined_data_1sw = pd.DataFrame()
		for d in lv1_data_to_include:
			combined_data_1sw = combineData(combined_data_1sw, data[sw][d], sw, d)
		for dir, d in lv2_data_to_include:
			#The biweekly tables are different lengths for each sw system.
			#Use only the most recent 90 (the minimum) biweekly periods.
			#Renumber as 1-90 before combining.
			#Need to keep classname column for the extra classname removal process in combineData()
			if 'biweekly' in dir:
				d_new = pd.DataFrame()
				d_new = pd.concat([data[sw][dir][d]['classname '], data[sw][dir][d][data[sw][dir][d].columns[-90:]]], axis=1)
				d_new.columns = range(91)
				d_new = d_new.rename(columns={0: 'classname '})
				combined_data_1sw = combineData(combined_data_1sw, d_new, sw, dir + '_' + d)
			else:
				combined_data_1sw = combineData(combined_data_1sw, data[sw][dir][d], sw, dir + '_' + d)
		combined_data_1sw.index = data[sw]['bug-metrics']['classname ']
		
		labels = labels.append(labels_1sw)
		combined_data = combined_data.append(combined_data_1sw)
	
	"""
	def createFeature(bugs, nonTrivialBugs, majorBugs, criticalBugs, highPriorityBugs):
		'''
		Function to create the values for the custom severity feature.
		This function is the same as createLabels, but returns numeric values instead of strings.
		'''
		if (bugs + nonTrivialBugs + majorBugs + criticalBugs + highPriorityBugs == 0): #this is assuming there aren't "negative" bug counts
			return 0
		elif (bugs != 0) and (nonTrivialBugs + majorBugs + criticalBugs + highPriorityBugs == 0):
			return 1
		else:
			return 2
	
	
	#add custom severity feature
	combined_data['custom_severity'] = combined_data.apply(lambda row: createFeature(row['bug-metrics_ numberOfBugsFoundUntil: '],
	                                                                                 row['bug-metrics_ numberOfNonTrivialBugsFoundUntil: '],
																					 row['bug-metrics_ numberOfMajorBugsFoundUntil: '],
																					 row['bug-metrics_ numberOfCriticalBugsFoundUntil: '],
																					 row['bug-metrics_ numberOfHighPriorityBugsFoundUntil: ']), axis=1)
	print(combined_data['custom_severity'].value_counts())
	"""
	
	#combined_data.to_csv('test.csv', index=False)
	
	return combined_data, labels