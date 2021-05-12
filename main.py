import joblib
from numpy import save
import pandas as pd
from joblib import dump, load
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

#print(X)
#print(y)

# 80% dei dati per allenamento, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# training con decision tree
new_model = input("Do you want to create a new model? [Y/n] ")
if new_model.lower() == 'y': 
    classifier = DecisionTreeClassifier()
    while True:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy is ", accuracy)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(y_pred)
        # salvataggio del modello
        if accuracy >= 0.8:
            print("Model saved\n")
            dump(classifier, 'model.joblib') 
            break
else:
    print("Model loaded\n")
    classifier = load('model.joblib')

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


