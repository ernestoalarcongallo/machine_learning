import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd
import random

# Example of euclidian distance manual calculation:
# note anyway we are using numpy.
def euclidian_distance_of_two_points(plot1,plot2):
    distance = 0
    for i, _ in enumerate(plot1):
        distance += (plot1[i]-plot2[i])**2
    return sqrt(distance)

#print(euclidian_distance_of_two_points([1,2],[5,6]))

def plot_2D_dataset(dataset, feature):
    # not compacted formula:
    # for i in dataset:
    #     for ii in dataset[i]:
    #         plt.scatter(ii[0], ii[1], s=100, color=i)
    #         plt.show()

    # compacted formula:
    [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
    if feature is not None:
        plt.scatter(feature[0], feature[1], s=100, color='g')
    plt.show()

def k_nearest_neighbors(dataset, new_feature, k=3):
    if len(dataset) >= k:
        warnings.warn('K is set to a value less than total groups to vote.')

    distances = []

    for theClass in dataset:
        for features in dataset[theClass]:
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(new_feature))
            distances.append([euclidian_distance, theClass])
    
    votes = [i[1] for i in sorted(distances) [:k]] # we only pic the k number distances
    
    # Counter(votes).most_common(1) = [(group, how_many)]
    # Counter(votes).most_common(1)[0] = (group, how_many)
    # Counter(votes).most_common(1)[0][1] = how_many
    # So you want to get that the classifier finds five times this class if k=5
    confidence = Counter(votes).most_common(1)[0][1] / k

    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result, confidence


def load_data(fileName):
    df = pd.read_csv(fileName)
    return df

def preprocess_data(df):
    df.replace('?',-99999, inplace=True)
    df.drop(['id'],1,inplace=True)
    full_data = df.astype(float).values.tolist() # make it a list of lists and avoid some columns threated as string
    random.shuffle(full_data)
    return full_data

def split_data(full_data, test_size=0.2, theClasses=[]):
    trainSet = {}
    testSet = {}

    for theClass in theClasses:
        trainSet[theClass] = []
        testSet[theClass] = []

    trainData = full_data[:-int(test_size*len(full_data))]
    testData = full_data[-int(test_size*len(full_data)):]

    for data in trainData:
        trainSet[data[-1]].append(data[:-1])

    for data in testData:
        testSet[data[-1]].append(data[:-1])

    return trainSet, testSet

#############################################
# Execute the algorithm with an example data:
#############################################

example_dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
example_new_feature = [5,7]

result = k_nearest_neighbors(example_dataset, example_new_feature, k=3)
print('\nThe prediction for the new feature {} is that belongs to class: {}\n'.format(example_new_feature,result))
#plot_2D_dataset(example_dataset, example_new_feature)

#################################################################
# Execute the algorithm compared to sklearn loading the UCI data:
#################################################################

df = load_data("breast-cancer-wisconsin.data")
full_data = preprocess_data(df)
trainSet, testSet = split_data(full_data, theClasses=[2,4])

correct, total = 0, 0

for theClass in testSet:
    for data in testSet[theClass]:
        prediction, confidence = k_nearest_neighbors(trainSet, data, k=5) #by default sci-kit chooses k=5, so as we.
        if theClass == prediction:
            correct += 1
        else:
            print('The confidence for an incorrect classification was: {}'.format(confidence))
        total += 1
    
print('ACCURACY: {}'.format(correct/total))