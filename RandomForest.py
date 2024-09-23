import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class RandomForest: 
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.dataset_without_duplicates = self.original_dataset.drop_duplicates()
        self.rf_classifier = None
        self.vectorizer = TfidfVectorizer(max_features=500)  

    def find_x_and_y(self, dataset):
        dataset.rename(columns={'inform im looking for a moderately priced restaurant that serves': 'utterance content'}, inplace=True)
        x = self.vectorizer.fit_transform(dataset['utterance content'])  
        y = dataset['dialog act']  

        return x, y
    
    def train_and_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(x_train, y_train)
        y_pred = self.rf_classifier.predict(x_test)

        return y_test, y_pred

    def perform_random_forest(self):
        # Dataset with duplicates
        x, y = self.find_x_and_y(self.original_dataset) 
        y_test, y_pred = self.train_and_test(x, y)        
        print("Accuracy with duplicates:", accuracy_score(y_test, y_pred))

        # Dataset without duplicates
        x, y = self.find_x_and_y(self.dataset_without_duplicates)        
        y_test, y_pred = self.train_and_test(x, y)
        print("Accuracy without duplicates:", accuracy_score(y_test, y_pred))








