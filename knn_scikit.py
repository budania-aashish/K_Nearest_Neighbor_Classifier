import pandas as pd
from sklearn.neighbors import KNeighborsClassifier



def knn(test,arr):
	for i in range(len(arr)):
		print("The value of k is",arr[i])
		neigh = KNeighborsClassifier(n_neighbors=arr[i])
		#selecting first four columns from the datasets
		neigh.fit(data.iloc[:,0:4], data['Name'])

		# Predicted class
		print("result is ",neigh.predict(test))

		#printing neighbors based on the values of the k 
		print("neighbors are", neigh.kneighbors(test)[1])
	

#importing datasets into data  
data = pd.read_csv("data_knn.csv")

#array for the values of the k 
arr=[1,3,5,7,9,11,13,15,17]

testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

knn(test,arr)