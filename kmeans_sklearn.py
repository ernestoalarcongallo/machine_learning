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
import numpy as np
import os

#######################################
### DATA PRINTING/LOADING FUNCTIONS ###
#######################################

def prepare_data(fileName):
    """Loads and prepares the data in fileName.
    - The file must be CSV
    - All fields must be integers
    - First row is skipped
    - Last column is ground truth (i.e. class)
    - Standardization is performed to all data"""

    # Load CSV file
    inputData=np.loadtxt(open(fileName), delimiter=",", skiprows=1, dtype='int')
    # Retrieve the samples (all but the last column)
    sampleData=inputData[:,:-1]
    # Get the classes (last column)
    groundTruth=inputData[:,-1]
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

def example():
    """Example"""
    # Configure numpy print options
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    # Load data. This assumes a CSV where all values are numeric, the last
    # column denotes the classes (ground truth) and all sample data is
    # pre-processed using standardization.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = dir_path + '/' + 'small.csv' 
    [sampleData,groundTruth]=prepare_data(file)
    # Execute K-means with 2 clusters and pre-defined centroids.
    kMeansOut=KMeans(n_clusters=2,init=sampleData[0:2,:],max_iter=25).fit(sampleData)
    # Print results
    print_final_results(kMeansOut.labels_,kMeansOut.cluster_centers_,groundTruth)

example()