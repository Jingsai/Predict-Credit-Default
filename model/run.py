import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import stats
import data
from LinearRegressionStat import LinearRegressionStat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=1000)

trainX=data.load_data('train.csv')
trainY=data.load_data('trainLabels.csv')
testX=data.load_data('test.csv')

# plt.boxplot(trainX)
# plt.show()

## collinear
# c=np.corrcoef(np.hstack((trainX,trainY[:,None])).T)
# c[c>0.99]=0
# print c
# print np.max(c), np.min(c)

# #LinearRegressionStat(trainX,trainY,coef_=True,t_=True,p_=True,F_=True,R2_=True)
# regr=linear_model.LinearRegression()
# regr.fit(trainX,trainY)
# trainY_hat=regr.predict(trainX)
# print "linear regression score\n", regr.score(trainX,trainY)
# for i in range(len(trainY_hat)):
#     if i>0.5:
#         trainY_hat[i]=1
#     else:
#         trainY_hat[i]=0
# print  "linear regression adjusted rate\n", sum(trainY==trainY_hat)/np.float(trainY.shape[0])

# lda=LinearDiscriminantAnalysis(solver="svd")
# lda.fit(trainX, trainY)
# # trainY_hat=lda.predict(trainX)
# print  "lda score\n", lda.score(trainX,trainY)
# # rate=sum(trainY==trainY_hat)/np.float(trainY.shape[0])  # same as score

# logit=linear_model.LogisticRegression(C=1e10)
# logit.fit(trainX,trainY)
# print  "logistic regression score\n", logit.score(trainX,trainY)

# # split data into train and test sets
# seed = 7
# test_size = 0.60
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainX, trainY, test_size=test_size, random_state=seed)
# print X_train.shape, X_test.shape
# # fit model no training data
# model = xgboost.XGBClassifier()
# model.fit(X_train, y_train)
# # make predictions for test data
# predictions = model.predict(X_test)
# #predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.base import BaseEstimator,ClassifierMixin

X_test = pd.read_csv('test.csv', header=None).as_matrix()
y = pd.read_csv('trainLabels.csv', header=None)[0].as_matrix()
X = pd.read_csv('train.csv', header=None).as_matrix()

def kde_plot(x):
    from scipy.stats.kde import gaussian_kde
    kde = gaussian_kde(x)
    positions = np.linspace(x.min(), x.max())
    smoothed = kde(positions)
    plt.plot(positions, smoothed)

def qq_plot(x):
    from scipy.stats import probplot
    probplot(x, dist='norm', plot=plt)

# pca2 = PCA(n_components=2, whiten=True)
# pca2.fit(np.r_[X, X_test])
# X_pca = pca2.transform(X)
i0 = np.argwhere(y == 0)[:, 0]
i1 = np.argwhere(y == 1)[:, 0]
# X0 = X_pca[i0, :]
# X1 = X_pca[i1, :]
# plt.plot(X0[:, 0], X0[:, 1], 'ro')
# plt.plot(X1[:, 0], X1[:, 1], 'b*')

pca = PCA(whiten=True)
X_all = pca.fit_transform(np.r_[X, X_test])
X=pca.transform(X)

#print pca.explained_variance_ratio_
#fig=plt.figure(figsize=(10, 8), dpi= 800)
#qq_plot(X_all[:,0])

class PcaGmm(BaseEstimator):
    def __init__(self, X_all,
                 pca_components = 11, gmm_components = 4,
                 covariance_type = "full", min_covar = 0.1,
                 gamma = 0, C = 1.0):

        self.pca_components = pca_components
        self.gmm_components = gmm_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.gamma = gamma
        self.C = C
        self.X_all = X_all

#         X_all = X_all[:, :pca_components]
##         print pca_components,X_all.shape
#         self.gmm = GMM(n_components = gmm_components,
#                        covariance_type = covariance_type,
#                        min_covar = min_covar)
#         self.gmm.fit(X_all)

    def fit(self, X, y):
        self.X_all = self.X_all[:, :self.pca_components]
        self.gmm = GMM(n_components = self.gmm_components,
                       covariance_type = self.covariance_type,
                       min_covar = self.min_covar)
        self.gmm.fit(self.X_all)
        X = X[:, :self.pca_components]
        #print X.shape
        X = self.gmm.predict_proba(X)
        self.svm = SVC(C = self.C, gamma = self.gamma)
        self.svm.fit(X, y)

    def predict(self, X):
        X = X[:, :self.pca_components]
        return self.svm.predict(self.gmm.predict_proba(X))

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def transform(self, X, y = None):
        X = X[:, :self.pca_components]
        return self.gmm.predict_proba(X)

    def __str__(self):
        return "PCA(%d)-GMM(%d, %s, %f)-SVM(C=%f, gamma=%f)" % (self.pca_components, self.gmm_components,
                                                                self.covariance_type, self.min_covar,
                                                                self.C, self.gamma)

tuned_parameters = { 'pca_components': [11,12],
                     'gmm_components': [4,5,6],
                     'gamma': [1,0.3], #[1, 0.3, 0.27, 0.26, 0.25, 0.24, 0.23, 0.2, 0.15, 0.1, 0.05, 1e-2, 1e-3, 1e-4],
                     'C': [0.5,0.9]} #[0.5, 0.9, 1, 1.1, 1.2, 2]}
cvk = cross_validation.StratifiedKFold(y, n_folds=2)
gv = GridSearchCV(PcaGmm(X_all), tuned_parameters, cv=cvk, n_jobs=1 , verbose=1)
gv.fit(X,y)
print gv.best_estimator_

# clf = PcaGmm(X_all[:1000,], 12, 4, 'full', 0, gamma = .6, C = 0.3)
# #print clf. __str__
# X_t = clf.transform(pca.transform(X))
# print X_t.shape
# print clf
# X0 = X_t[i0, :]
# X1 = X_t[i1, :]
# pca2 = PCA(n_components=2)
# pca2.fit(np.r_[X0, X1])
# X0 = pca2.transform(X0)
# X1 = pca2.transform(X1)
# plt.plot(X0[:, 0], X0[:, 1], 'ro')
# plt.plot(X1[:, 0], X1[:, 1], 'b*')