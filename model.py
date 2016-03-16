from sklearn import utils
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import cluster
from sklearn import feature_selection
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import *
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.externals.joblib import Memory
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn import decomposition
from sklearn import metrics

def prediction_model(method='RandomForest', **kwargs):
    method = method.lower()
    if method == "randomforest": return ensemble.RandomForestRegressor(**kwargs)
    if method == "decisiontree": return tree.DecisionTreeRegressor(**kwargs)
    if method == "lasso": return linear_model.Lasso(**kwargs)
    if method == "lassocv": return linear_model.LassoCV(**kwargs)
    if method == "multitasklasso": return linear_model.MultiTaskLasso(**kwargs)
    if method == "multitasklassocv": return linear_model.MultiTaskLassoCV(**kwargs)
    if method == "elasticnet": return linear_model.ElasticNet(**kwargs)
    if method == "elasticnetcv": return linear_model.ElasticNetCV(**kwargs)
    if method == "multitaskelasticnet": return linear_model.MultiTaskElasticNet(**kwargs)
    if method == "multitaskelasticnetcv": return linear_model.MultiTaskElasticNetCV(**kwargs)
    if method == "lassolars": return linear_model.LassoLars(**kwargs)
    if method == "lassolarscv": return linear_model.LassoLarsCV(**kwargs)
    if method == "lassolarsic": return linear_model.LassoLarsIC(**kwargs)
    if method == "lars": return linear_model.Lars(**kwargs)
    if method == "larscv": return linear_model.LarsCV(**kwargs)
    if method == "ridge": return linear_model.Ridge(**kwargs)
    if method == "ridgecv": return linear_model.RidgeCV(**kwargs)
    if method == "svr": return svm.SVR(**kwargs)
    if method == "logisticregression": return linear_model.LogisticRegression(**kwargs)
    if method == "logisticregressioncv": return linear_model.LogisticRegressionCV(**kwargs)
    if method == "sgd": return linear_model.SGDRegressor(**kwargs)
    if method == "gradientboosting": return ensemble.GradientBoostingRegressor(**kwargs)
    if method == "perceptron": return linear_model.Perceptron(**kwargs)
    if method == "bayesianridge": return linear_model.BayesianRidge(**kwargs)
    if method == "ardregression": return linear_model.ARDRegression(**kwargs)
    if method == "passiveaggressiveregressor": return linear_model.PassiveAggressiveRegressor(**kwargs)
    if method == "ransacregressor": return linear_model.RANSACRegressor(linear_model.ElasticNetCV(**kwargs))
    if method == "theilsenregressor": return linear_model.TheilSenRegressor(**kwargs)
    if method == "gaussianprocess": return GaussianProcess(**kwargs)
    if method == "omp": return linear_model.OrthogonalMatchingPursuit(**kwargs)
    if method == "rfecv": return RFECV(estimator=linear_model.Lasso(), step=1, cv=10, scoring=myscore)

    return None

def model_mre(model, training_indep, training_dep, testing_indep, testing_dep):
    model.fit(training_indep, training_dep)
    predictions = [float(x) for x in model.predict(testing_indep)]
    mre = []
    for i, j in zip(testing_dep, predictions):
        if i != 0:
            mre.append(abs(i - j) / float(i))  # abs(original - predicted)/original
        else:
            if i==j: mre.append(0)

    return mre
