'''
BPD_classification.py: classification script for the bug prediction data set.
Comment out different elements of the lists and write different for loops to
run different experiments.
'''

from time import time
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.decomposition import PCA
from umap import UMAP
from BPD_preprocess import preprocessData
from imblearn.over_sampling import RandomOverSampler

def printMetrics(y_true, y_pred, y_score, name, time):
	'''
	Prints evaluation metrics for a classification model in a tab-delimited table format.
	'''
	acc = accuracy_score(y_true, y_pred)
	#auc = roc_auc_score(y_true, y_score) #binary
	auc = roc_auc_score(y_true, y_score, multi_class='ovr') #multiclass
	p_triv = precision_score(y_true, y_pred, labels=['triv'], average=None, zero_division=0)
	r_triv = recall_score(y_true, y_pred, labels=['triv'], average=None, zero_division=0)
	p_ntrv = precision_score(y_true, y_pred, labels=['ntrv'], average=None, zero_division=0)
	r_ntrv = recall_score(y_true, y_pred, labels=['ntrv'], average=None, zero_division=0)
	print('%s\t%.2f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f' % (name, time, acc, auc, p_triv, r_triv, p_ntrv, r_ntrv))
	print(confusion_matrix(y_true, y_pred, labels=['none', 'triv', 'ntrv']))

SEED = 23

data, labels = preprocessData()
X = data.to_numpy()
y = labels.to_numpy()
	
classifiers = {
#'DecisionTree ': DecisionTreeClassifier(),
#'GaussianNB   ': GaussianNB(),
#'KNeighbors   ': KNeighborsClassifier(),
#'SGD          ': SGDClassifier(), #no probabilities, can't use in multiclass roc_auc
#'SVC          ': SVC(probability=True), #slow (~2m)
#'RandomForest ': RandomForestClassifier(random_state=SEED),
#'AdaBoost     ': AdaBoostClassifier(),
'GradientBoost': GradientBoostingClassifier(), #slooow (~6m)...although it does give best results
#'MLP          ': MLPClassifier(), #tried hidden_layer_sizes=(500, 100, 20). barely improved but took 5x longer
}

dim_reducers = {
'PCA-2': PCA(n_components=2),
'PCA-5': PCA(n_components=5),
'PCA-20': PCA(n_components=20),
'PCA-50': PCA(n_components=50),
'UMAP-2': UMAP(n_components=2, random_state=SEED),
'UMAP-5': UMAP(n_components=5, random_state=SEED),
'UMAP-20': UMAP(n_components=20, random_state=SEED),
'UMAP-50': UMAP(n_components=50, random_state=SEED),
}

feature_subsets = [
'' #all features
#'bug-metrics',
#'change-metrics',
#'complexity-code-change',
#'single-version-ck-oo',
#'biweekly-ck-values',
#'biweekly-oo-values',
#'churn',
#'entropy',
#'custom_severity',
]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=SEED)
#Maybe also try stratified KFold to account for imbalanced class distribution
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print('model\t\ttime\tacc\troc_auc\tp_triv\tr_triv\tp_ntrv\tr_ntrv')
for cl_name, cl in classifiers.items():
	for fs in feature_subsets:
		ti = time()
		col_subset = [col for col in data if col.startswith(fs)]
		data_subset = data[col_subset]
		X_subset = data_subset.to_numpy()
		y_true = np.array([])
		y_pred = np.array([])
		#y_score = np.array([]) #binary
		y_score = np.empty([0, 3]) #multiclass
		for train_index, test_index in kf.split(X_subset, y):
			X_train, X_test = X_subset[train_index], X_subset[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			X_train, y_train = RandomOverSampler(random_state=SEED).fit_resample(X_train, y_train)
			
			y_true = np.append(y_true, y_test)
			model = cl.fit(X_train, y_train)
			y_pred = np.append(y_pred, model.predict(X_test))
			#y_score = np.append(y_score, model.decision_function(X_test)) #binary
			y_score = np.append(y_score, model.predict_proba(X_test), axis=0) #multiclass
		tf = time()
		printMetrics(y_true, y_pred, y_score, fs, tf - ti)