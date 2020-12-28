from preprocess import *
import sklearn.metrics as metrics
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# confusion matrix
conf_matrix = metrics.confusion_matrix (target_test, pred)
print("Confusion matrix:\n", conf_matrix)

class_names = data['quality'].unique()
df_conf = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
sns.heatmap(df_conf, annot=True,  fmt='g')
plt.title("Confusion matrix: KNN")
plt.xlabel("False")
plt.ylabel("True")
plt.savefig("./graphs/confusion_matrix_knn.jpg")
plt.show()
