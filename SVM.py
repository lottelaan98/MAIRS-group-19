import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVMClassifier:
    """
    Support Vector Machine classifier for dialog act classification.
    """

    def __init__(self, dataset):
        self.original_dataset = dataset
        self.dataset_without_duplicates = self.original_dataset.drop_duplicates(subset=['utterance content'])
        self.svm_classifier = None
        self.vectorizer = TfidfVectorizer(max_features=500)

    def find_x_and_y(self, dataset):
        x = self.vectorizer.fit_transform(dataset['utterance content'])
        y = dataset['dialog act']
        return x, y

    def train_and_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        self.svm_classifier = SVC()
        self.svm_classifier.fit(x_train, y_train)
        y_pred = self.svm_classifier.predict(x_test)
        return y_test, y_pred

    def report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    def difficult_cases(self):
        keywords = {
            "ack": [
                "alright", "nice", "looking forward", "looking forward to it", "perfect",
                "excellent", "splendid", "very good", "couldn't be better", "can't be better",
                "great", "that's great", "wonderful", "fantastic", "awesome", "brilliant"
            ],
            "confirm": [
                "and the address", "and the location", "and the food type", "and the cuisine",
                "and the area", "and the pricerange", "and the reservation details", "and the date",
                "and the parking possibilities", "and the payment method"
            ],
            "repeat": [
                "sorry", "repeat that please", "could you repeat", "can you repeat", 
                "previous", "previous please", "try again", "pardon", "i missed that",
                "one more time", "say that again", "i didn't catch that", "rewind", 
                "go over that again", "what did you say", "please go over that again", 
                "can you clarify", "would you mind repeating"
            ]
        }

        data = [(key, phrase) for key, phrases in keywords.items() for phrase in phrases]
        df = pd.DataFrame(data, columns=["dialog act", "utterance content"])
        return df

    def perform_svm(self):
        x, y = self.find_x_and_y(self.dataset_without_duplicates)
        y_test, y_pred = self.train_and_test(x, y)
        class_report = self.report(y_test, y_pred)
        print(class_report)

    def perform_svm_difficult(self):
        df = self.difficult_cases()
        x, y = self.find_x_and_y(self.dataset_without_duplicates)
        x_vec_diff, y_vec_diff = self.find_x_and_y(df)
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.15, random_state=42)
        self.svm_classifier.fit(x_train, y_train)
        y_pred = self.svm_classifier.predict(x_vec_diff)
        class_report = self.report(y_vec_diff, y_pred)
        print(class_report)