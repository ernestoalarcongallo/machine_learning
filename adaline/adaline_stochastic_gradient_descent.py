import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class AdalineSGD(object):
  """
  - Adaline using Stochastic Gradient descent
  - This code pretends to be Python agnistic, that's why we inherit from object.
  
  Parameters
  ----------
  eta: float
    Learning rate, between 0.0 and 1.0.
  n_iter: int
    Number of passes over the trainning dataset.
  shuffle: bool (default: True)
    Shuffles the training data every epoch if activated
  random_state: int
    A random number generator seed for initialising the weights. 
    Weight should never be zero at the init time.
  
  Attributes
  ----------
  w_: 1d-array
    Weights after fitting.
  cost_: list
    Sum-of-squares cost function value 
    averaged over all training samples in each epoch.
  """

  def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
    self.eta = eta
    self.n_iter = n_iter
    self.w_initialized = False
    self.shuffle = shuffle
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data

    Parameters
    ----------
    X: {array like}, shape = [n_samples, n_features]
      Training vectors
    y: {array like}, shape = [n_samples]
      Target values

    Returns
    -------
    self: object

    """

    self._initialize_weights(X.shape[1])
    self.cost_ = []

    for _ in range(self.n_iter):
      if self.shuffle:
        X, y = self._shuffle(X, y)
      cost = []
      for xi, target in zip(X, y):
        cost.append(self._update_weights(xi, target))
      avg_cost = sum(cost) / len(y)
      self.cost_.append(avg_cost)
    return self

  def partial_fit(self, X, y):
    """ Fit training data without reinitializing the weights """
    if not self.w_initialized:
      self._initialize_weights(X.shape[1])
    if y.ravel().shape[0] > 1:
      for xi, target in zip(X, y):
        self._update_weights(xi, target)
    else:
      self._update_weights(X, y)
    return self

  def _shuffle(self, X, y):
    """ Shuffle the training data """
    r = self.rgen.permutation(len(y))
    return X[r], y[r]

  def _initialize_weights(self, m):
    self.rgen = np.random.RandomState(self.random_state)
    self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
    self.w_initialized = True

  def _update_weights(self, xi, target):
    """Apply Adaline learning rule to update the weights (the cost function) """
    output = self.activation(self.net_input(xi))
    error = (target - output)
    self.w_[1:] += self.eta * xi.T.dot(error)
    self.w_[0] += self.eta * error.sum()
    cost = error**2 / 2
    return cost

  def net_input(self, X):
    """Calculate net input, w[0] is the biass, w[1:] are the wj weights"""
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def activation(self, X):
    """In Adaline the activation function is the X function itself."""
    return X

  def predict(self, X):
    """Predict class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

def main():
  df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
  print(df.tail())

  # Select setosa and versicolor
  # |Sepal lenght|Sepal widht|Petal lenght|Petal widht|
  y = df.iloc[0:100, 4].values
  y = np.where(y == 'Iris-setosa', -1, 1)

  # Select sepal and petal lenght
  X = df.iloc[0:100, [0, 2]].values

  # Standardize the features to help the classifier
  X_std = np.copy(X)
  X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
  X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

  # Create the classifier
  adaline = AdalineSGD(n_iter=10, eta=0.01).fit(X_std, y)

  plot_decision_regions(X_std, y, clf=adaline)
  plt.title('Adaline - Gradient Descent')
  plt.xlabel('sepal length')
  plt.ylabel('petal length')
  plt.legend(loc='upper left')
  plt.show()

  plt.plot(range(1, len(adaline.cost_) + 1), adaline.cost_, marker='o')
  plt.xlabel('Iterations')
  plt.ylabel('Sum-squared-error')
  plt.show()

main()
