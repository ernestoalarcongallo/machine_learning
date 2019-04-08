from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
from random import randint

def load_data(fileName):
    df = pd.read_csv(fileName)
    return df

def load_data_without_headers(fileName):
    df = pd.read_csv(fileName, header=None)
    return df

def preprocess_data(df):
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

def get_sampleData_and_groundTruth(df):
    features = np.array(df.iloc[:,:-1])
    labels = np.array(df.iloc[:,-1])
    return features, labels

def prepare_data(fileName):
    df = load_data(fileName)
    preprocess_data(df)
    sampleData, groundTruth = get_sampleData_and_groundTruth(df)
    theMeans=np.mean(sampleData,0)
    theStd=np.std(sampleData,0)
    return [np.divide((sampleData-theMeans),theStd),groundTruth]

def print_results(theClasses,theCentroids):
    foundClasses=np.unique(theClasses)
    numFoundClasses=foundClasses.shape[0]
    formattedClasses=[]
    for i in range(numFoundClasses):
        cIndex=[j+1 for j,x in enumerate(theClasses) if x==foundClasses[i]]
        #print (' - CLASS '+str(int(foundClasses[i]))+' MEMBERS        : '+str(cIndex))
        print (' - CLASS '+str(int(foundClasses[i]))+' CENTROID       : '+str(theCentroids[i,:]))

def get_accuracy(theClasses,groundTruth):
    trueEstimates=np.count_nonzero((theClasses-groundTruth)==0)
    totalEstimates=len(theClasses)
    return float(trueEstimates)/float(totalEstimates)

"""
Example:
theClasses =  np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
groundTruth = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
            Predicted                   Predicted
            unacc acc                   unacc acc
Real unacc  [4]   [0]       Real unacc  [T]   [F]
       acc  [2]   [4]              acc  [F]   [T]
Fails : 2 examples
"""
def confusion_matrix(theClasses, groundTruth, k):
  confusion_matrix = np.zeros((k,k))
  fails = 0
  for index, truth in enumerate(groundTruth):
    if truth == 0 and theClasses[index] == 0:
      confusion_matrix[0,0] = confusion_matrix[0,0] + 1
    elif truth == 0 and theClasses[index] == 1:
      confusion_matrix[0,1] = confusion_matrix[0,1] + 1
      fails += 1
    elif truth == 1 and theClasses[index] == 0:
      confusion_matrix[1,0] = confusion_matrix[1,0] + 1
      fails += 1
    elif truth == 1 and theClasses[index] == 1:
      confusion_matrix[1,1] = confusion_matrix[1,1] + 1      
      
  return confusion_matrix, fails

def print_final_results(theClasses,theCentroids,groundTruth, k=2):
    print('[K-MEANS OUTPUT]')
    print_results(theClasses,theCentroids)
    print('[K-MEANS PERFORMANCE]')
    print(' - ACCURACY               : '+str(100*get_accuracy(theClasses,groundTruth))+'%')
    theConfusionMatrix, fails = confusion_matrix(theClasses,groundTruth, k)
    print(' - CONFUSION MATRIX       :\n'+str(theConfusionMatrix))
    print(' - FAILS                  : '+str(fails))

def execute_kmeans(fileName, random=False):
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)

    if random:
        print("K-Means with custom random init")
        # Execute K-means with 2 clusters and custom RANDOM centroids.
        n=2
        init_sample=np.zeros((n, sampleData.shape[1]))
        for i in range(n):
            random_i = randint(0, sampleData.shape[0]-1)
            init_sample[i] = sampleData[random_i]
        kMeansOut=KMeans(n_clusters=n,init=init_sample,max_iter=25).fit(sampleData) # RANDOM
        print_final_results(kMeansOut.labels_,kMeansOut.cluster_centers_,groundTruth)
    else:
        print("K-Means")
        # Execute K-means with 2 clusters and PRE-DEFINED centroids.
        kMeansOut=KMeans(n_clusters=2,init=sampleData[0:2,:],max_iter=25).fit(sampleData)
        print_final_results(kMeansOut.labels_,kMeansOut.cluster_centers_,groundTruth)

def execute_kmeans_with_PCA(fileName):
    print("K-Means with PCA attributes")
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    df = load_data_without_headers(fileName)
    [sampleData, groundTruth] = get_sampleData_and_groundTruth(df)
    kMeansOut=KMeans(n_clusters=2,init=sampleData[0:2,:],max_iter=25).fit(sampleData)
    print_final_results(kMeansOut.labels_,kMeansOut.cluster_centers_,groundTruth)

def execute_kmeans_plus_plus(fileName):
    print("K-Means++")
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    kMeansOut=KMeans(n_clusters=2,init='k-means++',max_iter=25).fit(sampleData)
    print_final_results(kMeansOut.labels_, kMeansOut.cluster_centers_, groundTruth)

def execute_kmeans_random(fileName):
    print("K-Means random")
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    kMeansOut=KMeans(n_clusters=2,init='random',max_iter=25).fit(sampleData)
    print_final_results(kMeansOut.labels_, kMeansOut.cluster_centers_, groundTruth)

def execute_miniBatch_kmeans(fileName):
    print("Mini Batch K-Means ")
    np.set_printoptions(formatter={'float':lambda x: '%.2f'%x})
    [sampleData,groundTruth]=prepare_data(fileName)
    random_state = np.random.RandomState(0)
    miniBatchKMeans = MiniBatchKMeans(n_clusters=2, init='random', n_init=1, random_state=random_state).fit(sampleData)
    print_final_results(miniBatchKMeans.labels_, miniBatchKMeans.cluster_centers_, groundTruth)

execute_kmeans("car.csv")
execute_kmeans_with_PCA("pca_attributes_car.csv")
execute_kmeans_plus_plus("car.csv")
execute_kmeans_random("car.csv")
execute_miniBatch_kmeans("car.csv")