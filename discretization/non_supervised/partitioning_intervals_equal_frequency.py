import numpy as np

def equalFrequencyIntervals(X, k):
  """ Partitioning discretization intervals on equal frequency.

  Parameters
  ----------
  X: {array-like}, shape (n, 1)
    It's a plain array of dimension n x 1 (one characteristic). Should be a numpy array.
  k: integer.
    The desired number of intervals.
  
  Algorithm
  ---------
  1.- Sort array. In this case, to improve eficiency, we use the mergesort algorithm which has a O(n*log(n)) complexity.
  2.- Obtain frequency freq = n/k, where n is the number of examples and k is the desired number of intervals.
  3.- Execute the partitioning discretization intervals on equal frequency algorithm.

  Returns
  -------
  intervals: {array-like}, shape(k, freq)
    The desired intervals based on the frequency of the X example provided.

  """

  n = X.shape[0]
  i = 0
  intervals = []
  freq = int(round(n / k))

  X_sorted = np.sort(X, kind='mergesort')

  for _ in range(k):
    interval = []
    for _ in range (freq):
      interval.append(X_sorted[i])
      i += 1
    intervals.append(interval)

  return intervals

def main():
  X = np.array([55, 22, 27, 40, 28, 22, 28, 31, 27, 31, 31, 55])
  _ = equalFrequencyIntervals(X, k=3)

if __name__ == '__main__':
  main()