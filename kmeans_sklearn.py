###
# A sklearn K-means usage example.
# Please, note that most of the code is for loading/printing. The whole
# K-means functionality is embeded in sklearn KMeans.
###
# This code directly works with CSV files if the following conditions are met:
# - They only have numeric values.
# - Its last column have class (i.e. ground truth) data
# Also, take into account that the following assumptions have been performed:
# - Standardization for all attributes is acceptable
# - Classes can be assigned using euclidean distance
# - Centroids are computed as the mean of the class points
# - Initial centroids are good enough
# It is easy to adapt it to other configurations.
###
# Author   : Antoni Burguera Burguera
# Creation : 18-Feb-2018
###

from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import os
import pandas as pd
from random import randint

#######################################
### DATA PRINTING/LOADING FUNCTIONS ###
#######################################

def load_data(fileName):
    """Loads and prepares the data in fileName.
    - The file must be CSV"""

    df = pd.read_csv(fileName)
    return df

def load_data_without_headers(fileName):
    df = pd.read_csv(fileName, header=None)
    return df

def preprocess_data(df):
    """Preprocess the input data according the criteria observed in the csv
    - We will convert into integers the data provided
    - Safety: low->0 medium->1 hight->2
    - Class: unacc->0 acc->1"""

    col_safety = 'SAFETY'
    col_class = 'CLASS'

    for index, row in enumerate(df[col_safety]):
        if row == 'low':
            df.iloc[index, df.columns.get_loc(col_safety)] = 0
        if row == 'med':
            df.iloc[index, df.columns.get_loc(col_safety)] = 1
        if row == 'high':
            df.iloc[index, df.columns.get_loc(col_safety)] = 2
        
    for index, row in enumerate(df[col_class]):
        if row == 'acc':
            df.iloc[index, df.columns.get_loc(col_class)] = 1
        if row == 'unacc':
            df.iloc[index, df.columns.get_loc(col_class)] = 0

def getFeaturesAndLabelsFrom(df):
    """Separe the features from the labels and return it"""

    features = np.array(df.iloc[:,:-1])
    labels = np.array(df.iloc[:,-1])
    return features, labels

def prepare_data(fileName):
    """Loads and prepares the data in fileName.
    - The file must be CSV
    - All fields must be integers
    - First row is skipped
    - Last column is ground truth (i.e. class)
    - Standardization is performed to all data"""

    # Load CSV file
    df = load_data(fileName)
    preprocess_data(df)
    sampleData, groundTruth = getFeaturesAndLabelsFrom(df)
    # Get means and standard deviations
    theMeans=np.mean(sampleData,0)
    theStd=np.std(sampleData,0)
    # Return the standardized data and the ground truth
    return [np.divide((sampleData-theMeans),theStd),groundTruth]

def print_results(theClasses,theCentroids):
    """Prints the class members and centroids"""
    # Get the existing classes
    foundClasses=np.unique(theClasses)
    numFoundClasses=foundClasses.shape[0]
    # Allocte space
    formattedClasses=[]
    # For each found class
    for i in range(numFoundClasses):
        # Get the sample indexes+1 to print class members
        cIndex=[j+1 for j,x in enumerate(theClasses) if x==foundClasses[i]]
        # Print the members and the centroids
        print (' - CLASS '+str(int(foundClasses[i]))+' MEMBERS        : '+str(cIndex))
        print (' - CLASS '+str(int(foundClasses[i]))+' CENTROID       : '+str(theCentroids[i,:]))

def get_accuracy(theClasses,groundTruth):
    """Computes the accuracy has the ratio between true estimates and total estimates"""
    trueEstimates=np.count_nonzero((theClasses-groundTruth)==0)
    totalEstimates=len(theClasses)
    return float(trueEstimates)/float(totalEstimates)

def print_final_results(theClasses,theCentroids,groundTruth):
    """Prints K-Means output and performance"""
    print('[K-MEANS OUTPUT]')
    print_results(theClasses,theCentroids)
    print('[K-MEANS PERFORMANCE]')
    print(' - ACCURACY               : '+str(100*get_accuracy(theClasses,groundTruth))+'%')


#####################
### USAGE EXAMPLE ###
#####################

def example(fileName):
    """Example"""
    # Configure numpy print options
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    # Load data. This assumes a CSV where all values are numeric, the last
    # column denotes the classes (ground truth) and all sample data is
    # pre-processed using standardization.
    
    [sampleData,groundTruth]=prepare_data(fileName)
    # Execute K-means with 2 clusters and pre-defined centroids.
    # FOR RANDOM:
    n=2
    init_sample=np.zeros((n, sampleData.shape[1]))
    for i in range(n):
        random_i = randint(0, sampleData.shape[0]-1)
        init_sample[i] = sampleData[random_i]
    #kMeansOut=KMeans(n_clusters=n,init=init_sample,max_iter=25).fit(sampleData) # RANDOM
    
    # FOR THE TWO FIRST
    kMeansOut=KMeans(n_clusters=2,init=sampleData[0:2,:],max_iter=25).fit(sampleData)
    
    # Print results
    print_final_results(kMeansOut.labels_,kMeansOut.cluster_centers_,groundTruth)

def example_with_PCA(fileName):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    df = load_data_without_headers(fileName)
    [sampleData, groundTruth] = getFeaturesAndLabelsFrom(df)
    kMeansOut=KMeans(n_clusters=2,init=sampleData[0:2,:],max_iter=25).fit(sampleData)
    print_final_results(kMeansOut.labels_,kMeansOut.cluster_centers_,groundTruth)

def example_KMeans_plus(fileName):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    kMeansOut=KMeans(n_clusters=2,init='k-means++',max_iter=25).fit(sampleData)
    print_final_results(kMeansOut.labels_, kMeansOut.cluster_centers_, groundTruth)

def example_KMeans_random(fileName):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    kMeansOut=KMeans(n_clusters=2,init='random',max_iter=25).fit(sampleData)
    print_final_results(kMeansOut.labels_, kMeansOut.cluster_centers_, groundTruth)

def example_MiniBatch_KMeans(fileName):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    random_state = np.random.RandomState(0)
    miniBatchKMeans = MiniBatchKMeans(n_clusters=2, init='random', n_init=1, random_state=random_state).fit(sampleData)
    print_final_results(miniBatchKMeans.labels_, miniBatchKMeans.cluster_centers_, groundTruth)
    
# example("small.csv")
# example_with_PCA("pcaAttributes.csv")

#example("car.csv")
#example_with_PCA("pca_attributes_car.csv")
example_KMeans_plus("car.csv")
example_KMeans_random("car.csv")
#example_MiniBatch_KMeans("car.csv")