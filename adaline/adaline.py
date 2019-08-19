    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class Adaline(object):
  """
  - This code pretends to be Python agnistic, that's why we inherit from object.
  
  Parameters
  ----------
  eta: float
    Learning rate, between 0.0 and 1.0.
  n_iter: int
    Number of passes over the trainning dataset.
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

  def __init__(self, eta=0.01, n_iter=10, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data
    
    Parameters
    ----------
    X: {array like}, shape = [n_samples, n_features]
    y: {array like}, shape = [n_samples]

    Returns
    -------
    self: object

    """

    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
    self.cost_ = []

    for _ in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (errors**2).sum() / 2.0
      self.cost_.append(cost)
    return self

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

  _ , ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

  # Too big learning rate (eta=0.01)
  adaline1 = Adaline(n_iter=10, eta=0.01).fit(X, y)
  ax[0].plot(range(1, len(adaline1.cost_) + 1), np.log10(adaline1.cost_), marker='o')
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('log(Sum-squared-error)')
  ax[0].set_title('Adaline - Learning rate 0.01')

  # Too small learning rate (eta=0.0001)
  adaline2 = Adaline(n_iter=10, eta=0.0001).fit(X, y)
  ax[1].plot(range(1, len(adaline2.cost_) + 1), adaline2.cost_, marker='o')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Sum-squared-error')
  ax[1].set_title('Adaline - Learning rate 0.0001')

  plt.show()

  # Now let's standardize the features to help the classifier
  X_std = np.copy(X)
  X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
  X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

  adaline3 = Adaline(n_iter=10, eta=0.01).fit(X_std, y)
  plot_decision_regions(X_std, y, clf=adaline3)
  plt.title('Adaline - Gradient Descent')
  plt.xlabel('sepal length')
  plt.ylabel('petal length')
  plt.legend(loc='upper left')
  plt.show()

  plt.plot(range(1, len(adaline3.cost_) + 1), adaline3.cost_, marker='o')
  plt.xlabel('Iterations')
  plt.ylabel('Sum-squared-error')
  plt.show()

main()
