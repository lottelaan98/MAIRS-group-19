# Restaurant recommendation system
from RandomForest import RandomForest
import StateTransitions
import Levenshtein
from StateTransitions import keywords_1
import pandas as pd
import time
from Baseline2 import Baseline2
import random
import uuid

##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################

file_path_restaurants = "C:\\Users\\certj\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\Part 2\\MAIRS-group-19\\MAIRS-group-19\\restaurant_info2.csv"

file_path_dialog = "C:\\Users\\certj\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\Part 2\\MAIRS-group-19\\MAIRS-group-19\\dialog_acts.dat"

allow_dialog_restarts: bool = True
use_delay: bool = False
output_in_caps: bool = False
use_baseline_as_classifier: bool = False

# Give a random transparency level. If you want to test it, manually set it to transparency of 0, 1 or 2
transparency_level = 2  # random.randint(0, 2)
"""
    Transparency levels are about the recommendation of the system:
    0 = System utterance consists of the restaurant only.
    1 = System utterance consists of the currently selected restaurant and why we choose that
    2 = System utterance consists of the currently selected restaurant, why we choose that and why we did not choose the others.

"""


def load_data() -> pd.DataFrame:
    # Load the data into a DataFrame
    df = pd.read_csv(file_path_dialog, header=None)
    # Split each row into 'dialog act' and 'utterance content'
    df["dialog act"] = df[0].apply(lambda x: x.split(" ", 1)[0].lower())
    df["utterance content"] = df[0].apply(lambda x: x.split(" ", 1)[1].lower())
    df = df.drop(columns=[0])
    return df


class SystemDialog:
    def __init__(self):
        self.dataset = load_data()
        # Store the random_forest instance here
        self.random_forest = self.train_random_forest_classifier()
        # Access vectorizer after initialization
        self.vectorizer = self.random_forest.vectorizer
        self.state = StateTransitions.State(file_path_restaurants, transparency_level)
        self.acts = StateTransitions.Dialog_Acts()
        self.turn_index = 0
        self.baseline2 = Baseline2(self.dataset)

    def train_random_forest_classifier(self):
        """
        This function is called in the beginning, in order to have a classifier to classify the user utterances in 15 different dialog acts
        """
        print("One moment please. We are training our classifier.")
        random_forest = RandomForest(self.dataset)
        random_forest.perform_random_forest()

        return random_forest

    def classify(self, user_input) -> str:
        """
        Input is a user utterance. Output is the predicted dialog act (i.e. class) of this user utterance.
        """

        def correct_sentence(sentence):
            corrected_sentence = []
            words = sentence.split()  # Split sentence into words

            # Iterate over each word in the sentence
            for word in words:
                corrected_word = word
                min_overall_distance = 2

                # Check the word against all categories
                for category in keywords_1:
                    for keyword in keywords_1[category]:
                        # Compute Levenshtein distance between word and keyword
                        distance = Levenshtein.distance(word, keyword)

                        # If a closer match is found, update the corrected word
                        if distance < min_overall_distance:
                            min_overall_distance = distance
                            corrected_word = keyword

                corrected_sentence.append(corrected_word)

            return " ".join(corrected_sentence)

        def classify_user_input(user_input) -> str:
            if use_baseline_as_classifier:
                predicted_class: str = self.baseline2.classify(user_input)
            else:
                # Transform input to match the trained model
                preprocessed_input = self.vectorizer.transform([user_input.lower()])
                predicted_class: str = self.random_forest.rf_classifier.predict(
                    preprocessed_input
                )[0]
            return predicted_class

        predicted_class = classify_user_input(user_input)

        if predicted_class == "null" or predicted_class == "unknown":
            corrected_user_input = correct_sentence(user_input)

            predicted_class = classify_user_input(corrected_user_input)

        # Preferably, the (baseline2) classifier doesn't predict this class. If it does, make it 'null'
        if predicted_class == "unknown":
            predicted_class == "null"

        return predicted_class

    def state_transition(self, predicted_class, user_input):
        self.turn_index += 1
        if predicted_class == "ack":
            return self.acts.ack(self.state)
        elif predicted_class == "affirm":
            return self.acts.affirm(self.state)
        elif predicted_class == "bye":
            return self.acts.bye(self.state)
        elif predicted_class == "confirm":
            return self.acts.confirm(self.state, user_input)
        elif predicted_class == "deny":
            return self.acts.deny(self.state, user_input)
        elif predicted_class == "hello":
            return self.acts.hello(self.state)
        elif predicted_class == "inform":
            return self.acts.inform(self.state, user_input)
        elif predicted_class == "negate":
            return self.acts.negate(self.state, user_input)
        elif predicted_class == "null":
            return self.acts.null(self.state, user_input)
        elif predicted_class == "repeat":
            return self.acts.repeat(self.state)
        elif predicted_class == "reqalts":
            return self.acts.reqalts(self.state, user_input)
        elif predicted_class == "reqmore":
            return self.acts.reqmore(self.state)
        elif predicted_class == "request":
            return self.acts.request(self.state, user_input)
        elif predicted_class == "restart":
            return self.acts.restart(self.state, allow_dialog_restarts)
        else:
            return self.acts.thankyou(self.state, user_input)

    def dialog_system(self):
        self.state.last_system_utterance = "System: Hello, welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?"

        # Print the welcome text
        if output_in_caps:
            self.state.last_system_utterance = self.state.last_system_utterance.upper()

        print(self.state.last_system_utterance)
        # Create dialog_log to be able to save the dialog later
        dialog_log = [self.state.last_system_utterance]
        dialog_log.append(f"State = {self.state.current_state}")

        # Do the rest of the dialog
        while self.state.current_state != "End":
            user_input = input("Me: ").lower()
            dialog_log.append(f"Me: {user_input}")

            # Predict class and perform actions based on this.
            predicted_class = self.classify(user_input)
            system_utterance = self.state_transition(predicted_class, user_input)
            self.state.last_system_utterance = system_utterance

            # Insert delay if use_delay is True
            if use_delay:
                time.sleep(1)

            # Print the system utterance to the console.
            if output_in_caps:
                system_utterance = system_utterance.upper()

            print("System: ", system_utterance)
            dialog_log.append(f"System: {system_utterance}")
            dialog_log.append(f"State = {self.state.current_state}")

        self.save_dialog(dialog_log)

    def save_dialog(self, dialog_log):
        dialog_id = uuid.uuid4()
        filename = f"dialog_{dialog_id}.txt"

        # Open het bestand en schrijf de dialoog erin
        with open(filename, "w") as f:
            f.write(f"Dialog ID: {dialog_id}\n")
            f.write(f"Turn Index: {self.turn_index}\n")
            f.write(f"Transparency Level: {transparency_level}\n\n")

            # Schrijf alle dialoogregels naar het bestand
            for line in dialog_log:
                f.write(line + "\n")

        print(f"Dialog saved to {filename}")


system_dialog = SystemDialog()
system_dialog.dialog_system()
