###
# A basic PCA implementation
###
# This code directly works with CSV files if the following conditions are met:
# - They only have numeric values.
# - Its last column have class (i.e. ground truth) data
# Also, take into account that the following assumptions have been performed:
# - Standardization for all attributes is acceptable
###
# Author   : Antoni Burguera Burguera
# Creation : 18-Feb-2018
###

import numpy as np

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

#############################
### PCA RELATED FUNCTIONS ###
#############################

def pca(sampleData):
    """Aplies PCA to the provided data. Outputs the sorted eigenvalues
    and eigenvectors."""

    # Standardize data
    sampleData-=np.mean(sampleData,0)
    sampleData/=np.std(sampleData,0)
    # Get the covariance
    numSamples=sampleData.shape[0]
    theCovariance=np.dot(sampleData.T,sampleData)/numSamples
    # Get eigenvalues and eigenvectors
    [eigenValues,eigenVectors]=np.linalg.eigh(theCovariance)
    # Sort them descending
    iSort=np.argsort(eigenValues)[::-1]
    [eigenValues,eigenVectors] = [eigenValues[iSort],eigenVectors[:,iSort]]
    #outData=np.dot(sampleData,eigenVectors)
    # Return the eigenvalues and vectors
    return eigenValues,eigenVectors

def get_explained_variance(eigenValues):
    """Outputs the cumulative variance"""

    # The explained variance for each component is its eigenvalue divided
    # by the sum of all eigenvectors. This code returns the cumulative
    # sum of explained variances.
    return np.cumsum(np.divide(eigenValues,np.sum(eigenValues)))

#####################
### USAGE EXAMPLE ###
#####################

def example():
    """Loads example data (see prepare_data for format). Prints cumulative
    variance and the projected attributes preserving 95% or more variance"""

    # Load data
    [sampleData,groundTruth]=prepare_data('small.csv')
    # Do PCA
    [eigenValues,eigenVectors]=pca(sampleData)
    # Get cumulative explained variances
    explainedVariances=get_explained_variance(eigenValues)
    # Get the number of vectors to reach 95%
    numVectors=np.argmax(explainedVariances>.95)
    # Build the new attribute set
    projectedAttributes=np.dot(sampleData,eigenVectors[:,:numVectors+1])
    # Print results
    print('[PCA OUTPUT]')
    print(' - CUMULATIVE VARIANCES: '+str(explainedVariances))
    print(' - VECTORS REQUIRED    : '+str(numVectors+1))
    print(' - PROJECTED ATTRIBUTES:\n'+str(projectedAttributes))

example()