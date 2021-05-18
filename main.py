from numpy import save
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

new_model = input("Do you want to create a new model? [Y/n] ")
if new_model.lower() == 'y':
    path_train = input("Enter .csv training source path: ")
    dataset = pd.read_csv(path_train)
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
    # training con random forest
    classifier = RandomForestClassifier()
    wanted_acc = float(input("Type wanted accuracy: "))
    while True:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy is ", accuracy)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(y_pred)
        # salvataggio del modello
        if accuracy >= wanted_acc:
            dump(classifier, 'model.joblib')
            print("Model saved\n")
            break
else:
    classifier = load('model.joblib')
    print("Model loaded\n")

# prediction
path_topredict = input("Enter .csv to predict path: ")
print("\t---RESULTS---")
# carico file in variabile da utilizzare e variabile di output
to_pr = pd.read_csv(path_topredict)
csv_res = pd.read_csv(path_topredict)
# cancello le colonne inutili
to_pr.drop('objType', axis=1, inplace=True)
csv_res.drop('objType', axis=1, inplace=True)

names = to_pr['object']
to_pr.drop('object', axis=1, inplace=True)
to_pr.drop('collection', axis=1, inplace=True)
pred = classifier.predict(to_pr)

# df contenente i valori predictati
pred_df = pd.DataFrame(pred)

# concateno per la stampa del csv di output
res = pd.concat([names, pred_df], axis=1)
res.rename({ 0 : 'objType'}, axis=1, inplace=True)
print(res)

csv_res = pd.concat([csv_res, pred_df], axis=1)
csv_res.rename({ 0 : 'objType'}, axis=1, inplace=True)
print(csv_res.head(5))
res_name = input("Type result .csv name: ")
csv_res.to_csv(res_name, index=False)




