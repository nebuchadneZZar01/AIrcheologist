import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('mongialino.csv')
print(dataset.head(5))

dataset.drop('object', axis=1, inplace=True)
dataset.drop('collection', axis=1, inplace=True)

#target
y = dataset['objType']
#features
X = dataset.drop('objType', axis=1)

print(X)
print(y)

# 80% dei dati per allenamento, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train)
print(y_train)

# training con decision tree
print("DECISION TREE")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred,))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(y_pred)

# prediction
print("\t---RESULTS---")
pr = pd.read_csv('mongialino_topredict.csv')
names = pr['object']
pr.drop('object', axis=1, inplace=True)
pr.drop('collection', axis=1, inplace=True)
pred = classifier.predict(pr)

pred_df = pd.DataFrame(pred)

# result
res = pd.concat([names, pred_df], axis=1)
res.rename({ 0 : 'objType'}, axis=1, inplace=True)
print(res)


