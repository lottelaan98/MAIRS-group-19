import pandas as pd
from Baseline1 import Baseline1
from Baseline2 import Baseline2
from SVM import SVM
from RandomForest import RandomForest
from sklearn.metrics import classification_report

##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################

file_path_dialog = "YOUR_FILE_PATH\\dialog_acts.dat"


def load_data_dialog() -> pd.DataFrame:
    # Load the data into a DataFrame
    df_dialog = pd.read_csv(file_path_dialog, header=None)
    # Split each row into 'dialog act' and 'utterance content'
    df_dialog["dialog act"] = df_dialog[0].apply(lambda x: x.split(" ", 1)[0].lower())
    df_dialog["utterance content"] = df_dialog[0].apply(
        lambda x: x.split(" ", 1)[1].lower()
    )
    df_dialog = df_dialog.drop(columns=[0])
    return df_dialog


def without_duplicates(data):
    dataset_without_duplicates = data.drop_duplicates(subset=["utterance content"])
    return dataset_without_duplicates


class ClassifierAnalyzer:
    def __init__(self, data):
        self.data = data
        self.data_without_duplicates = self.without_duplicates(data)
        self.classifiers = {
            "Baseline1": Baseline1(data),
            "Baseline2": Baseline2(data),
            "RandomForest": RandomForest(data),
            "SVM": SVM(data),
        }

    def without_duplicates(self, data):
        return data.drop_duplicates(subset=["utterance content"])

    def analyze_performance(self):
        for name, classifier in self.classifiers.items():
            # Evaluate with duplicates
            x, y = classifier.find_x_and_y(self.data)
            y_test, y_pred = classifier.train_and_test(x, y)
            print(
                f"{name}: Report of duplicated data:\n{classification_report(y_test, y_pred)}"
            )

            # Evaluate without duplicates
            x, y = classifier.find_x_and_y(self.data_without_duplicates)
            y_test, y_pred = classifier.train_and_test(x, y)
            print(
                f"{name}: Report of deduplicated data:\n{classification_report(y_test, y_pred)}"
            )

    def hard_utterance_classification(self):
        for name, classifier in self.classifiers.items():
            x, y = classifier.find_x_and_y(self.data)
            y_test, y_pred = classifier.train_and_test(x, y)
            results = pd.DataFrame(
                {
                    "utterance content": self.data["utterance content"].iloc[
                        y_test.index
                    ],
                    "actual dialog act": y_test,
                    "predicted dialog act": y_pred,
                }
            )

            # Identify incorrect classifications
            incorrect_classifications = results[
                results["actual dialog act"] != results["predicted dialog act"]
            ]
            print(
                f"{name}: Incorrectly classified utterances with duplicates:\n",
                incorrect_classifications[
                    ["utterance content", "actual dialog act", "predicted dialog act"]
                ],
            )

            # Save the incorrect classifications to a CSV file
            incorrect_file_name = (
                f"{name}_incorrect_classifications_with_duplicates.csv"
            )
            incorrect_classifications.to_csv(incorrect_file_name, index=False)
            print(
                f"Saved incorrectly classified utterances with dupcliates to {incorrect_file_name}"
            )

        for name, classifier in self.classifiers.items():
            x, y = classifier.find_x_and_y(self.data_without_duplicates)
            y_test, y_pred = classifier.train_and_test(x, y)
            results = pd.DataFrame(
                {
                    "utterance content": self.data["utterance content"].iloc[
                        y_test.index
                    ],
                    "actual dialog act": y_test,
                    "predicted dialog act": y_pred,
                }
            )

            # Identify incorrect classifications
            incorrect_classifications = results[
                results["actual dialog act"] != results["predicted dialog act"]
            ]
            print(
                f"{name}: Incorrectly classified utterances without duplicates:\n",
                incorrect_classifications[
                    ["utterance content", "actual dialog act", "predicted dialog act"]
                ],
            )

            # Save the incorrect classifications to a CSV file
            incorrect_file_name = (
                f"{name}_incorrect_classifications_without_duplicates.csv"
            )
            incorrect_classifications.to_csv(incorrect_file_name, index=False)
            print(
                f"Saved incorrectly classified utterances without duplicates to {incorrect_file_name}"
            )


def analyse_data():
    df_dialog: pd.DataFrame = load_data_dialog()

    number_of_dialogs = len(df_dialog)
    print("Number of dialogs is:", number_of_dialogs)

    frequency_dialog_act = df_dialog["dialog act"].value_counts()
    print(frequency_dialog_act)

    print("Number of different dialog acts is", len(frequency_dialog_act))

    df_without_duplicates = df_dialog.drop_duplicates(subset=["utterance content"])
    number_deduplicates = len(df_without_duplicates)
    print("Number of deduplicates:", number_deduplicates)

    number_of_duplicate_utterances = number_of_dialogs - number_deduplicates
    percentage_duplicated = (number_of_duplicate_utterances / number_of_dialogs) * 100
    print("Percentage of duplicated utterances ", percentage_duplicated, "%")


df_dialog = load_data_dialog()

classifieranalyzer = ClassifierAnalyzer(df_dialog)
classifieranalyzer.analyze_performance()
classifieranalyzer.hard_utterance_classification()

analyse_data()
