import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVM:
    """
    Support Vector Machine classifies utterances based on the most common class
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_without_duplicates = self.dataset.drop_duplicates()
    
    def vectorize(self, dataset):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(dataset['utterance content'])
        y = dataset['dialog act']
        return X, y

    def split_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        return x_train, x_test, y_train, y_test

    def fit_svm(self, x_train, y_train):
        svm = SVC()
        svm.fit(x_train, y_train)
        return svm
        
    def make_prediction(self, x_test, svm):
        y_pred = svm.predict(x_test)
        return y_pred

    def report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        return report
    
    def print_wrong_predictions(self, y_test, y_pred, dialog_act=None):
        # Convert y_test to an array (if it's a pandas Series)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test

        # Iterate through predictions and compare them with true labels
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                # Check if dialog_act filter is provided, and if the true label matches the filter
                if dialog_act is None or y_test[i] == dialog_act:
                    # Find the original row in the dataset using the index of x_test
                    original_row = self.dataset.iloc[i]
                    
                    # Print the relevant row along with true and predicted labels
                    print(f"Index {i}: Original utterance: {original_row['utterance content']}")
                    print(f"True label = {y_test[i]}, Predicted label = {y_pred[i]}")
                    print("-" * 80)
                    
    def difficult_cases(self):
        keywords = {
                'ack': ['alright', 'nice', 'looking forward', 'looking forward to it', 'perfect', 'excellent', 'splendid', 
                        'very good', 'couldn\'t be better', 'can\'t be better', 'great', 'that\'s great', 'wonderful', 'fantastic', 'awesome',
                        'brilliant'],
                'confirm': ['and the address', 'and the location', 'and the food type', 'and the cuisine', 'and the area', 'and the pricerange',
                            'and the reservation details', 'and the date', 'and the parking possibilities', 'and the payment method' 
                            ],
                'repeat': ['sorry', 'repeat that please', 'could you repeat',
                        'can you repeat' ,'could you repeat that please',
                        'previous', 'previous please', 'can repeat', 'try again',
                        'pardon', 'excuse me', 'i missed that', 'one more time', 'one more time please',
                        'say that again', 'repeat it please', 'i didn\'t catch that', 'i did not catch that',
                        'rewind', 'rewind that please', 'go over that again', 'can you go over that once more',
                        'what did you say', 'i missed what you said', 'could you reiterate', 'please go over that again',
                        'can you clarify', 'would you mind repeating' 
                        ],
            }
        
        data = [(key, phrase) for key, phrases in keywords.items() for phrase in phrases]
        
        df = pd.DataFrame(data, columns=['dialog act', 'utterance content'])
        return df['dialog act'], df['utterance content']

    def perform_svm(self):
            x, y = self.vectorize(self.dataset)
            x_train, x_test, y_train, y_test = self.split_data(x, y)
            fitted_svm = self.fit_svm(x_train, y_train)
            # test it for difficult cases
            # y_diff, x_diff = self.difficult_cases()
            # y_pred = self.make_prediction(x_test, fitted_svm)
            y_pred = self.make_prediction(x_test, fitted_svm)

            # Create a report of the results
            class_report = self.report(y_test, y_pred)
            print(class_report)
            
            
    
    
