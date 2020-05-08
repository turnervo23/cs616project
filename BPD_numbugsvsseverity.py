'''
BPD_numbugsvsseverity.py: this just creates a csv to be used in Excel
to check correlation. It probably could've all been done in Excel but
I don't know how to do it.
'''

import pandas as pd
import os

raw_data = pd.concat([pd.read_csv('../data/eclipse/bug-metrics.csv'),
                      pd.read_csv('../data/equinox/bug-metrics.csv'),
					  pd.read_csv('../data/lucene/bug-metrics.csv'),
					  pd.read_csv('../data/mylyn/bug-metrics.csv'),
					  pd.read_csv('../data/pde/bug-metrics.csv')])

def createLabels(bugs, nonTrivialBugs, majorBugs, criticalBugs, highPriorityBugs):
	'''
	Create labels from the 5 bugs columns.
	Numbered in order of increasing severity.
	'''
	if (bugs + nonTrivialBugs + majorBugs + criticalBugs + highPriorityBugs == 0): #this is assuming there aren't "negative" bug counts
		return 0
	elif (bugs != 0) and (nonTrivialBugs + majorBugs + criticalBugs + highPriorityBugs == 0):
		return 1
	else:
		return 2

data = raw_data.apply(lambda row: createLabels(row[' bugs '], row[' nonTrivialBugs '], row[' majorBugs '], row[' criticalBugs '], row[' highPriorityBugs ']), axis=1)
data.name = 'severity'
data = pd.concat([raw_data[' bugs '], data], axis=1)

data = data[data['severity'] != 0]
data.to_csv('../results/numbugsvsseverity.csv', index=False)