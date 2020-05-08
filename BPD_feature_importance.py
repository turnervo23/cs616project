'''
BPD_feature_importance.py: alternate classification script. This is only
separate from the BPD_classification.py because I didn't like having to keep 
commenting out the feature importance related code whenever I wasn't using it.
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

def printMetrics(y_true, y_pred, y_score, name, time):
	acc = accuracy_score(y_true, y_pred)
	auc = roc_auc_score(y_true, y_score, multi_class='ovr')
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

feature_subsets = [
'', #all
'bug-metrics',
'change-metrics',
'complexity-code-change',
'single-version-ck-oo',
'biweekly-ck-values',
'biweekly-oo-values',
'churn',
'entropy',
]

'''
for fs in feature_subsets:
	print('---\n%s\n---' % fs)
	col_subset = [col for col in data if col.startswith(fs)]
	data_subset = data[col_subset]
	X_subset = data_subset.to_numpy()
	model = GradientBoostingClassifier().fit(X_subset, y)

	FI = {}
	for i in range(len(model.feature_importances_)):
		FI[data_subset.columns[i]] = model.feature_importances_[i]
	FI = sorted(FI.items(), key=lambda item: item[1], reverse=True)
	for i in FI:
		print(i[0] + ': ' + str(i[1]))
'''

model = GradientBoostingClassifier().fit(X, y)
FI = {}
for i in range(len(model.feature_importances_)):
	FI[data.columns[i]] = model.feature_importances_[i]
FI = sorted(FI.items(), key=lambda item: item[1], reverse=True)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print('model\t\ttime\tacc\troc_auc\tp_triv\tr_triv\tp_ntrv\tr_ntrv')
features_to_keep = []
for i in range(50):
	features_to_keep.append(FI[i][0])
data_subset = data[features_to_keep]
X_subset = data_subset.to_numpy()

estimator_counts = [100]#, 200, 500, 1000, 2000]
for n in estimator_counts:
	ti = time()
	y_true = np.array([])
	y_pred = np.array([])
	y_score = np.empty([0, 3])
	for train_index, test_index in kf.split(X_subset, y):
		X_train, X_test = X_subset[train_index], X_subset[test_index]
		y_train, y_test = y[train_index], y[test_index]
		y_true = np.append(y_true, y_test)
		model = GradientBoostingClassifier(n_estimators=n, class_weight='balanced').fit(X_train, y_train)
		y_pred = np.append(y_pred, model.predict(X_test))
		y_score = np.append(y_score, model.predict_proba(X_test), axis=0)
	tf = time()
	printMetrics(y_true, y_pred, y_score, str(n), tf - ti)