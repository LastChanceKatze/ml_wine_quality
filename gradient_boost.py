from sklearn.ensemble import GradientBoostingClassifier
import preprocess as pp
import evaluate as evl

# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data)

# random forest classifier
rfc = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.5, random_state=0)
model = rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
#

# cross validation score
evl.cross_val_eval(rfc, x_train, y_train, 5)

# accuracy, recall, precision, f1
evl.evaluate_model(y_test, y_pred)

# confusion matrix
evl.confusion_matrix(y_test, y_pred, data, True, "gradient_boost")
