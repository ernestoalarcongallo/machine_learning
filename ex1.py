import numpy as np
import pandas as pd

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

def getFeaturesAndLabelsFrom(df):
    """Separe the features from the labels and return it"""
    features = np.array(df.iloc[:,:-1])
    labels = np.array(df.iloc[:,-1])
    return features, labels

def standarize_features(features):
    """ Standarizes the data"""
    # Get means and standard deviations
    theMeans=np.mean(features,0)
    theStd=np.std(features,0)
    # Return the standardized data and the ground truth
    standarized_features = np.divide((features-theMeans),theStd)
    print('Standarized data:\n{}'.format(standarized_features))
    np.savetxt("standarized_data.csv", standarized_features, delimiter=",")
    return standarized_features

def pca(features):
    N = features.shape[0]
    factor = 1/N
    covariance = np.dot(features.T,features)/N
    print('covariance:\n{}'.format(covariance))
    np.savetxt("covariance_matrix.csv", covariance, delimiter=",")

    [eigenValues,eigenVectors] = np.linalg.eigh(covariance)
    # Sort them descending
    iSort = np.argsort(eigenValues)[::-1]
    [eigenValues,eigenVectors] = [eigenValues[iSort],eigenVectors[:,iSort]]
    print('Eigen Values:\n{}'.format(eigenValues))
    print('Eigen Vectors:\n{}'.format(eigenVectors))
    np.savetxt("eigenValues.csv", eigenValues, delimiter=",")
    np.savetxt("eigenVectors.csv", eigenVectors, delimiter=",")
    
    # Return the eigenvalues and vectors
    return eigenValues,eigenVectors

def get_explained_variance(eigenValues):
    """Outputs the cumulative variance"""

    # The explained variance for each component is its eigenvalue divided
    # by the sum of all eigenvectors. This code returns the cumulative
    # sum of explained variances.
    explainedVariances = np.cumsum(np.divide(eigenValues,np.sum(eigenValues)))
    print('Explained Variances:\n{}'.format(explainedVariances))
    np.savetxt("Explained Variances.csv", explainedVariances, delimiter=",")

    return explainedVariances

def execute():
    """Execute all the process step by step"""
    df = load_data('small.csv')
    preprocess_data(df)
    features, _ = getFeaturesAndLabelsFrom(df)
    features = standarize_features(features)
    eigenValues, eigenVectors = pca(features)
    explainedVariances = get_explained_variance(eigenValues)
    
    # Get the number of vectors to reach 95%
    numVectors=np.argmax(explainedVariances>.95)
    # Build the new attribute set
    print(eigenVectors[:,:numVectors])
    projectedAttributes=np.dot(features,eigenVectors[:,:numVectors+1])
    # Print results
    print('[PCA OUTPUT]')
    print(' - CUMULATIVE VARIANCES: '+str(explainedVariances))
    print(' - VECTORS REQUIRED    : '+str(numVectors+1))
    print(' - PROJECTED ATTRIBUTES:\n'+str(projectedAttributes))

execute()
