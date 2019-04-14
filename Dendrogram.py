import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

class Dendogram:
    def __init__(self, data):
        self.data = data

    def calculate_distances(self):
        self.distances = np.zeros((self.data.shape[0], self.data.shape[0]))

        for i, f1 in enumerate(self.data):
            for j,f2 in enumerate(self.data):
                self.distances[i, j] = np.linalg.norm(f1-f2)

        return self.distances

def load_data(fileName):
    """Loads and prepares the data in fileName.
    - The file must be CSV"""

    df = pd.read_csv(fileName)
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
    # Load CSV file
    df = load_data(fileName)
    preprocess_data(df)
    sampleData, groundTruth = getFeaturesAndLabelsFrom(df)
    # Get means and standard deviations
    theMeans=np.mean(sampleData,0)
    theStd=np.std(sampleData,0)
    # Return the standardized data and the ground truth
    return [np.divide((sampleData-theMeans),theStd),groundTruth]

def get_accuracy(theClasses,groundTruth):
    """Computes the accuracy has the ratio between true estimates and total estimates"""
    trueEstimates=np.count_nonzero((theClasses-groundTruth)==0)
    totalEstimates=len(theClasses)
    return float(trueEstimates)/float(totalEstimates)

def execute(fileName):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    features, _ = prepare_data(fileName)
    dendogram = Dendogram(features)
    distances = dendogram.calculate_distances()
    #print(np.around(distances, decimals=2))
    np.savetxt("distances.csv", distances, fmt='%.2f', delimiter=",")

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

def cluster_example(fileName):
    [sampleData,groundTruth]=prepare_data(fileName)
    #print_final_results(kMeansOut.labels_, kMeansOut.cluster_centers_, groundTruth)

    # ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    #                    400., 754., 564., 138., 219., 869., 669.])
    
    
    Z = hierarchy.linkage(sampleData, 'complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)

cluster_example("small.csv")