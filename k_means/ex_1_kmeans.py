import numpy as np
import pandas as pd
from random import randint

###################
# KMEANS ALGORITHM:
###################

class K_Means:
    def __init__(self, k=2, tol=0.00000001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}
        self.classesList = np.zeros((data.shape[0]))

        for i in range(self.k):
            # for random uncommend this lines
            #random_i = randint(0, data.shape[0]-1)
            #self.centroids[i] = data[random_i]
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for index, featureset in enumerate(data):
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                self.classesList[index] = classification
                        
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def plot_centroids(self):
        for key, values in self.centroids.items():
            print("\nThe centroid {}:\n{}".format(key, values))        

    def plot_classifications(self):
        for key, values in self.classifications.items():
            print("\nThe group {}:".format(key))
            for value in values:
                print(value)

    def centroids_list(self):
        centroids_list = []
        for key,value in self.centroids.items():
            centroids_list.append(value)
        return np.array(centroids_list)

#####################
# DATA PREPROCESSING:
#####################

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

##################################
# PRINTING RESULTS HELPER METHODS:
##################################

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

##################
# EXECUTE METHODS:
##################

def execute(fileName):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    print(sampleData)
    #k_means = K_Means(k=2)
    #k_means.fit(sampleData)
    #print_final_results(k_means.classesList, k_means.centroids_list(), groundTruth)

# def execute_with_PCA_attributes():
#     np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
#     df = load_data_without_headers("pcaAttributes.csv")
#     [sampleData,groundTruth]=getFeaturesAndLabelsFrom(df)
#     k_means = K_Means(k=2)
#     k_means.fit(sampleData)
#     print(k_means.classesList, k_means.centroids_list(), groundTruth)
#     print_final_results(k_means.classesList, k_means.centroids_list(), groundTruth)

execute('test.csv')