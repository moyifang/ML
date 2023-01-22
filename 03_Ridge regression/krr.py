import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer

class KRR():

  def __init__(self, alpha=np.logspace(-12, -6, 7), gamma=np.logspace(-4, -1, 1000)):

    self.scorer = make_scorer(mean_squared_error, greater_is_better=False)
    self.best_params_ = None
    self.best_score_ = None

    self.clf = GridSearchCV(KernelRidge(kernel='rbf', gamma=1.0), cv=KFold(n_splits=5, shuffle=True),
                            param_grid={"alpha": alpha, "gamma": gamma},
                            verbose=1, n_jobs=-1, scoring=self.scorer)

  def train(self, X, Y):

    self.clf.fit(X, Y)
    self.best_params_ = self.clf.best_params_
    self.best_score_ = self.clf.best_score_


  def predict(self, X):

    return self.clf.predict(X)
