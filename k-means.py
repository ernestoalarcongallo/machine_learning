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

def getClassesAndCentroidsFrom(df):
    """Separe the features from the labels and return it"""
    theClasses = np.array(df.iloc[:,:-1])
    theCentroids = np.array(df.iloc[:,-1])
    return theClasses, theCentroids

def print_results(theClasses,theCentroids):
    """Prints the class members and centroids"""
    # Get the existing classes
    foundClasses = np.unique(theClasses)
    print('Found Classses\n{}'.format(foundClasses))
    numFoundClasses = foundClasses.shape[0]
    print('numFoundClasses\n{}'.format(numFoundClasses))

    # Allocte space
    # formattedClasses=[]
    # For each found class
    # for i in range(numFoundClasses):
    #     # Get the sample indexes+1 to print class members
    #     cIndex=[j+1 for j,x in enumerate(theClasses) if x==foundClasses[i]]
    #     # Print the members and the centroids
    #     print (' - CLASS '+str(int(foundClasses[i]))+' MEMBERS        : '+str(cIndex))
    #     print (' - CLASS '+str(int(foundClasses[i]))+' CENTROID       : '+str(theCentroids[i]))

#def k_means(n_clusters=2,init,max_iter=25):

def distanceToMeans(df, theMeans):
    for index_col, aMean in enumerate(theMeans):
        for index_row, row in enumerate(df.iloc[:, index_col]):
            df.iloc[index_row, index_col] = row - aMean
    return df

def distancesInCluster(df):
    for column in df:
        #print(df[column])
        val = min(column, key=abs)
        for index, row in enumerate(df[column]):

def execute():
    df = load_data('small.csv')
    df = preprocess_data(df)
    df = standarize_features(df)
    df = distanceToMeans(df, np.array(df.iloc[0, :]))
    #theClasses, theCentroids = getClassesAndCentroidsFrom(df)
    #print_results(theClasses, theCentroids)
    #theClasses = distanceToMeans(theClasses, np.array(df.iloc[0, :]))
    distancesInCluster(df)

execute()