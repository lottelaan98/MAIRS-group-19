import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def read_data_as_dict(data):
    with open(f"./{data}") as myFile:
        formattedData = dict()
        for index, line in enumerate(myFile):
            try:
                clean_line = line.strip()
                data = clean_line.split(' ')
                dialog_act = data[0]
                text = data[1:]
                formattedData[index] =  {"dialog_act": dialog_act,
                                        "text": text}
            except:
                formattedData[index] = ""
                
    print(formattedData[0])
    return formattedData

def read_data_as_df():
    # read flash.dat to a list of lists
    df = pd.read_csv('dialog_acts.dat', header=None)
    return df

def form_baseline(data, prediction):
    total = len(data)
    counter = 0
    for element in data:
        if element["dialog_acts"] == prediction:
            counter += 1
        else:
            pass 
    
    accuracy = counter / total
    return accuracy

def process_data(data):
    data['labels'] = data[0].apply(lambda x: x.split()[0].lower())
    phase1 = data[0].apply(lambda x: ' '.join(x.split()[1:]))
    data['text'] = [i.lower() for i in phase1]
    return data['labels'], data['text']

def vectorize(labels, text):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    y = labels
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    return X_train, X_test, y_train, y_test

def fit_SVM(X_train, y_train):
    svm = SVC()
    svm.fit(X_train, y_train)
    return svm
    
def make_prediction(X_test, svm):
    y_pred = svm.predict(X_test)
    return y_pred

def report(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    return report

def SVM():
    df = read_data_as_df()
    labels, text = process_data(df)
    X, y = vectorize(labels, text)
    X_train, X_test, y_train, y_test = split_data(X, y)
    fitted_svm = fit_SVM(X_train, y_train)
    y_pred = make_prediction(X_test, fitted_svm)
    clas_rep = report(y_test, y_pred)
    print(clas_rep)
    
    
SVM()