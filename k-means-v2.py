import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy

def load_data(fileName):
    """Loads and prepares the data in fileName.
    - The file must be CSV"""

    df = pd.read_csv(fileName)
    return df

def preprocess_data(df):
    """Preprocess the input data according the criteria observed in the csv
    - We will convert into integers the data provided
    - Safety: low->1 medium->2 hight->3
    - Class: unacc->0 acc->1"""

    col_safety = 'SAFETY'
    col_class = 'CLASS'

    for index, row in enumerate(df[col_safety]):
        if row == 'low':
            df.iloc[index, df.columns.get_loc(col_safety)] = 1
        if row == 'med':
            df.iloc[index, df.columns.get_loc(col_safety)] = 2
        if row == 'high':
            df.iloc[index, df.columns.get_loc(col_safety)] = 3
        
    for index, row in enumerate(df[col_class]):
        if row == 'acc':
            df.iloc[index, df.columns.get_loc(col_class)] = 1
        if row == 'unacc':
            df.iloc[index, df.columns.get_loc(col_class)] = 0
    
    return df

def standarize_features(df):
    """ Standarizes the data"""
    features = np.array(df.iloc[:, :-1])
    # Get means and standard deviations
    theMeans=np.mean(features,0)
    theStd=np.std(features,0)
    # Return the standardized data and the ground truth
    features = np.divide((features-theMeans),theStd)
    df.iloc[:, :-1] = features
    return df

def separeClassesAndCentroidsFrom(df):
    """Separe the features from the labels and return it"""
    theClasses = np.array(df.iloc[:,:-1])
    theCentroids = np.array(df.iloc[:,-1])
    return theClasses, theCentroids

# def k_means(theClasses, theCategories, k, n, c):
#     print('K Means Executing... k={} n={} c={}'.format(k, n, c))
#     # Generate random centers, here we use sigma and mean to ensure it represent the whole data
#     mean = np.mean(theClasses, axis = 0)
#     std = np.std(theClasses, axis = 0)
#     centers = np.random.randn(k,c)*std + mean
#     #print('The Means\n{}\nThe STD\n{}\nThe Centers'.format(mean, std, centers))
#     #plotData(theClasses, theCategories, centers, colors=['orange', 'blue', 'green', 'yellow', 'red'])

#     centers_old = np.zeros(centers.shape) # to store old centers
#     centers_new = deepcopy(centers) # Store new centers

#     theClasses.shape
#     clusters = np.zeros(n)
#     distances = np.zeros((n,k))

#     error = np.linalg.norm(centers_new - centers_old)

#     # When, after an update, the estimate of that center stays the same, exit loop
#     while error != 0:
#         # Measure the distance to every center
#         for i in range(k):
#             distances[:,i] = np.linalg.norm(theClasses - centers[i], axis=1)
#         # Assign all training data to closest center
#         clusters = np.argmin(distances, axis = 1)
    
#         centers_old = deepcopy(centers_new)
#         # Calculate mean for every cluster and update the center
#         for i in range(k):
#             centers_new[i] = np.mean(theClasses[clusters == i], axis=0)
#         error = np.linalg.norm(centers_new - centers_old)

#     return centers_new

def k_mean2(theClasses, k, n, c):
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(theClasses, axis = 0)
    std = np.std(theClasses, axis = 0)
    centers = np.random.randn(k,c)*std + mean # as rows as k and as columns classes number
    #print('The centers\n{}'.format(centers))

    centers_old = np.zeros(centers.shape) # to store old centers
    centers_new = deepcopy(centers) # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n,k))

    error = np.subtract(centers_new, centers_old)

    # for i in range(k):
    #     print('->{}\n<-{}'.format(centers[:, i], theClasses[:, i]))
    #     distances = np.subtract(theClasses, centers[i,:])

    # for i in range(k):
    #     distances[:,i] = np.linalg.norm(theClasses - centers[i], axis=1)
    
    for index , column in enumerate(theClasses.T):
        distances[:, index] = np.subtract(column, theClasses[index,:])
    print('The distances\n{}'.format(distances))

    # distances = []
    # for i in range(k):
    #     for category, column in enumerate(centers.T):
    #         for center in column:
    #             diff = theClasses[:, category] - center
                


def plotData(theClasses, theCategories, centers, colors):
    # Plot the data and the centers generated as random
    n = theClasses.shape[0]
    for i in range(n):
        plt.scatter(theClasses[i, 0], theClasses[i,1], s=20, color = colors[int(theCategories[i])])
        plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
    plt.show()

def execute():
    df = load_data('small.csv')
    df = preprocess_data(df)
    df = standarize_features(df)
    theClasses, theCategories = separeClassesAndCentroidsFrom(df)
    print('The Classes:\n{}\nThe Categories:\n{}'.format(theClasses, theCategories))
    # Number of clusters will be initially be theClasses number
    k = theClasses.shape[1]
    # Number of training data
    n = theClasses.shape[0]
    # Number of features in the data
    c = theClasses.shape[1]
    # Execute K-Means
    #centers = k_means(theClasses, theCategories, k, n, c)
    #print(centers)
    k_mean2(theClasses, k, n, c)

execute()
