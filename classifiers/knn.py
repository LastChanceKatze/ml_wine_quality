from preprocess import *
import sklearn.metrics as metrics
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from descriptive_statistics import *
data=import_data()

data_train, data_test, target_train, target_test=preprocess(data)
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
#Train the algorithm
model=neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ", accuracy_score(target_test, pred))


