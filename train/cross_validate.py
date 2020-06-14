import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def get_timeseries_split(max_train_size = None, number_of_folds = 5):
  return TimeSeriesSplit(max_train_size=None, n_splits=5)


def cross_validate(model, X,y, batch_size, epochs, number_of_folds,callbacks):
  cv_scores = []
  timeseries_split = get_timeseries_split(number_of_folds = number_of_folds)
  for train_index, test_index in timeseries_split.split(X):
    n_model = model
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    n_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)
    scores = n_model.evaluate(X_test, y_test, callbacks=[callbacks[1]], verbose=1) # we only want the wandb callback..
    print("%s: %.5f%%" % (n_model.metrics_names[0], scores[0]))
    cv_scores.append(scores[0])
  return cv_scores, np.mean(cv_scores), np.std(cv_scores)