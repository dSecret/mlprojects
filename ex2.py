import random 
from scipy.spatial import distance

def euc(a,b):
	return distance.euclidean(a,b)

class Scrap():
	def fit(self,x,y):
		self.x=x
		self.y=y

		
	def predict(self,x_test):
		predictions=[]
		for row in x_test:
			label=self.closest(row)
			predictions.append(label)
		return predictions
	def closest(self,row):
		best_dist=euc(row,self.x[0])
		best_index=0
		for i in range(1,len(self.x)):
			dist=euc(row,self.x[i])
			if dist<best_dist:
				best_dist=dist
				best_index=i
		return self.y[best_index]






import sklearn
from  sklearn import datasets
iris=datasets.load_iris()

x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

from sklearn.neighbors import KNeighborsClassifier

my_classifier=Scrap()

my_classifier.fit(x_train,y_train)

predictions=my_classifier.predict(x_test)

# print (predictions)

from sklearn.metrics import accuracy_score

print (accuracy_score(y_test,predictions))


