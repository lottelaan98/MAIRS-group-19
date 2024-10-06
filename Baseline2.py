import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class Baseline2:
    """
    Baseline 2 classifies utterances based on previously defined key-words
    """

    df_keywords = {
        "request": [
            "what is",
            "address",
            "whats",
            "phone",
            "postcode",
            "post code",
            "price range",
            "type of food",
            "area",
        ],
        "thankyou": ["thank you", "okay"],
        "ack": ["okay", "um", "kay"],
        "affirm": ["yes", "right", "yea"],
        "bye": ["bye", "thank you"],
        "null": ["noise", "sil", "unintelligible", "cough"],
        "reqalts": [
            "how about",
            "how",
            "about",
            "else",
            "is there anything else",
            "anything else",
            "is there anything else",
        ],
        "inform": [
            "food",
            "restaurant",
            "town",
            "i dont care",
            "dont care",
            "it doesnt matter",
            "doesnt matter",
            "center",
            "north",
            "east",
            "south",
            "west",
            "any area",
            "any price range",
            "anything",
            "moderate",
            "cheap",
            "expensive",
            "any",
            "thai",
            "tailand",
            "lebanese",
            "italian",
            "italian food",
            "chinese",
            "chinese food",
            "spanish",
            "spanish food",
            "french",
            "portuguese",
            "korean",
            "turkish",
            "asian oriental",
            "indian",
            "vietnamese",
            "british food",
            "european",
            "mediterranean",
            "mediterranean food",
            "gastropub",
            "moderately",
        ],
        "deny": ["wrong", "want", "dont"],
        "negate": ["no"],
        "repeat": ["repeat", "back", "again"],
        "reqmore": ["more"],
        "restart": ["start over", "reset"],
        "hello": ["hi", "hello"],
        "confirm": ["it is", "it", "is"],
    }

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_without_duplicates = self.dataset.drop_duplicates(
            subset=["utterance content"]
        )
        self.vectorizer = TfidfVectorizer(max_features=500)

    def classify(self, sentence):
        """
        Finds the dialog act of a given sentence by looping over the keywords dictionary.
        Returns the predicted dialog act.
        """
        for dialog_act, keywords in self.df_keywords.items():
            for keyword in keywords:
                if keyword in sentence:
                    return dialog_act
        return "unknown"

    def evaluate(self, dataset):
        """
        Evaluates our keyword classifier.
        Returns the ratio of correctly predicted dialog acts to the total number of dialog acts.
        """
        correct = 0

        for _, row in dataset.iterrows():
            dialog_act = row["dialog act"]
            utterance = row["utterance content"]
            prediction = self.classify(utterance)
            if prediction == dialog_act:
                correct += 1

        return correct / len(dataset)

    def find_x_and_y(self, dataset):
        x = dataset["utterance content"]  # Keep original sentences for classification
        y = dataset["dialog act"]
        return x, y

    def train_and_test(self, x, y):
        y_pred = []
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

        # Classify each utterance in the test set
        for sentence in x_test:
            prediction = self.classify(sentence)  # Call the classify method
            y_pred.append(prediction)  # Append the prediction to the list

        return y_test, y_pred


# Sources
# https://sparkbyexamples.com/pandas/pandas-split-column/#:~:text=In%20Pandas%2C%20the%20apply(),to%20split%20into%20two%20columns.
# https://www.geeksforgeeks.org/reading-dat-file-in-python/
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
# https://stackoverflow.com/questions/21930035/how-to-write-help-description-text-for-python-functions
