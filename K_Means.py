import numpy as np
import pandas as pd

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i] # Set the first two centroids to start. TODO: set them random

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

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

def standarize_features(features):
    """ Standarizes the data"""
    # Get means and standard deviations
    theMeans=np.mean(features,0)
    theStd=np.std(features,0)
    # Return the standardized data and the ground truth
    standarized_features = np.divide((features-theMeans),theStd)
    np.savetxt("standarized_data.csv", standarized_features, delimiter=",")
    return standarized_features

def execute():
    """Execute all the process step by step"""
    df = load_data('small.csv')
    preprocess_data(df)
    features, _ = getFeaturesAndLabelsFrom(df)
    print("\nThe adapted data:\n{}\n".format(features))
    features = standarize_features(features)
    print("\nThe standarized data:\n{}\n".format(features))
    k_means = K_Means(k=4)
    k_means.fit(features)
    k_means.plot_centroids()
    k_means.plot_classifications()

execute()