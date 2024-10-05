# Restaurant recommendation system
import pandas as pd
from Baseline1 import Baseline1
from Baseline2 import Baseline2
from SVM import SVM
from RandomForest import RandomForest
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

file_path_dialog = "C:\\Users\\certj\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\dialog_acts.dat"


class Classification:
    def __init__(self, file_path_dialog):
        self.file_path_dialog = file_path_dialog

    def load_data(self) -> pd.DataFrame:
        # Location of the the data file. CHANGE THIS ACCORDING TO THE PATH ON YOUR OWN COMPUTER
        file_path = self.file_path_dialog

        # Load the data into a DataFrame
        df = pd.read_csv(file_path, header=None)
        # Split each row into 'dialog act' and 'utterance content'
        df["dialog act"] = df[0].apply(lambda x: x.split(" ", 1)[0].lower())
        df["utterance content"] = df[0].apply(lambda x: x.split(" ", 1)[1].lower())
        df = df.drop(columns=[0])
        return df

    def perform_classifications(self):
        df: pd.DataFrame = self.load_data()

        # Baseline1
        print("Start baseline 1...")
        baseline1 = Baseline1(df)
        most_common_class: str = baseline1.classify()
        print("Baseline 1 most common class = ", most_common_class)

        # Baseline2
        print("Start baseline 2...")
        baseline2 = Baseline2(df)
        accuracy_baseline2 = baseline2.evaluate(df)
        print("Baseline2 accuracy on current data set = ", accuracy_baseline2)

        # SVM
        print("Start Support Vector Machine...")
        svm = SVM(df)
        svm.perform_svm()

        # Random forest
        print("Start Random Forest classifier...")
        random_forest = RandomForest(df)
        random_forest.perform_random_forest()


# perform_classifications()
classification = Classification(
    "C:\\Users\\certj\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\dialog_acts.dat"
)
df = classification.load_data()
svm = SVM(df)
svm.perform_svm_difficult(5)
