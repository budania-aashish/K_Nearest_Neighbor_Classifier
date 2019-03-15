#python implementation of KNN algorithm 

# Importing libraries for mathematical functions 
import pandas as pd
import numpy as np
import math
import operator 


# Importing data 
data = pd.read_csv("data_knn.csv")
data.head() 

# Defining a function which calculates euclidean distance between two data points
# data is a vector and it can more than one attributes 
def euclideanDistance(data1, data2, length):
    distance = 0
    #data1 and data2 both of them are from training and testing sets 
    for x in range(length):
	## Euclidean method distance 
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

## defining the KNN model fror training set, testing instance and the value of k
def knn(trainingSet, testInstance, k):
 
    distances = {}
    sort = {}
 
    length = testInstance.shape[1] #lehgth is without label, so it's perfect for Euclidean distance
    #for all the rows of training data 
    for x in range(len(trainingSet)):
	#finding the euclidean distance with test instance 
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    #Sorting them on the basis of distance in ascending order 
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
     
    #a list of neighbors 
    neighbors = []
    
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}

    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)

testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)


arr = [1,3,5,7,9,11,13,15,17]

# Running KNN model
for i in range(len(arr)):
	result,neigh = knn(data, test, arr[i])
	print("The value of k is ",arr[i])
	# Predicted class
	print("result is ",result)
	print("neighbor is ",neigh)
