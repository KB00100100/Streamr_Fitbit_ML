import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

X = np.loadtxt('D:/wenjian/medical project/data/xlabel.csv', delimiter=',')
y = np.loadtxt('D:/wenjian/medical project/data/ylabel.csv',dtype='int32',delimiter=',')
X_2d = X[:, :]   #  ECG and RSP and RSA
y_2d = y[:]
min_max_scaler = preprocessing.MinMaxScaler()     # scale to 0-1
X_2d_minmax = min_max_scaler.fit_transform(X_2d)
C_range = np.logspace(-2, 10, 13)      # start  stop    num
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
X_2d_minmax, y_2d, test_size=0.2, random_state=None)     # split to train and test dataset
clf = OneVsRestClassifier(SVC(C=15,gamma=0.28,probability=True))   # parameters: three states and 35 sorted features
clf=clf.fit(X_train,y_train)
ytest_predict = clf.predict(X_test)
ytest_score = clf.predict_proba(X_test)
ytrain_predict = clf.predict(X_train)
ytrain_score = clf.predict_proba(X_train)

ytrainscore=clf.score(X_train,y_train)
ytestscore=clf.score(X_test,y_test)
############  scores for train datasets
train_report = metrics.classification_report(y_train[:],ytrain_predict[:])    # report : presion, recall, F1 score(P,R,F)
train_confusion = metrics.confusion_matrix(y_train,ytrain_predict)   # confusion score
train_accurency = metrics.accuracy_score(y_train,ytrain_predict)    # accurency score

print train_report
print "train_confusion metrix:"
print train_confusion
print "train_accurency:",train_accurency

############  scores for test datasets
test_report = metrics.classification_report(y_test[:],ytest_predict[:])    # report : presion, recall, F1 score(P,R,F)
test_confusion = metrics.confusion_matrix(y_test,ytest_predict)   # confusion score
test_accurency = metrics.accuracy_score(y_test,ytest_predict)    # accurency score

print test_report
print "test_confusion metrix:"
print test_confusion
print "test_accurency:",test_accurency

######score
print "test score:",clf.score(X_test,y_test)   # equal to test_accurency
