# Restaurant recommendation system
import pandas as pd
from Baseline1 import Baseline1
from Baseline2 import Baseline2
from SVM import SVM
from RandomForest import RandomForest
from RestaurantRecommendationClassification import load_data
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# This dictionary contains as keys all the possible dialog states. The values of these keys are the possible subsequent states.
dialog_state_dictionary = {
    "Welcome": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation"
    },
    "AskForMissingInfo": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation"
    },
    "AskUserForClarification": {
        "AskUserForClarification",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation"
    },
    "AskForConfirmation": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation"
    },
    "InformThatThereIsNoRestaurant": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation",
        "ProvideAlternativeSuggestion",
    },
    "GiveRestaurantRecommendation": {
        "AnswerAdditionalQuestion",
        "ProvideContactInformation",
    },
    "ProvideAlternativeSuggestion": {
        "AnswerAdditionalQuestion",
        "ProvideContactInformation",
    },
    "AnswerAdditionalQuestion": {
        "AnswerAdditionalQuestion",
        "ProvideContactInformation",
    },
    "ProvideContactInformation": {
        "AnswerAdditionalQuestion",
        "End",
    },
}

class SystemDialog:
    def __init__(self):
        self.current_state = "Welcome"
        self.rf_classifier = self.train_random_forest_classifier()

    def train_random_forest_classifier(self):
        """
        Trains the classifier in the Random Forest object, so that it can be used to classify user utterances.
        """
        df = load_data()

        # Random Forest 
        print('One moment please...')
        random_forest = RandomForest(df)
        random_forest.perform_random_forest()

        return random_forest.rf_classifier

    def new_state(self, user_input) -> str:
        """
        Changes the current_state and returns the system output
        """

        # output = requalts(food=european) | inform(=dont care) | inform(type=restaurant) | request(add, phone)
        return ""
    
    def process_user_input(self, input) -> str:
        # 1. classify user_input

        # 2.
        # IF DIALOG_ACT == REQUALTS | INFORM | REQUEST | NEGATE
        #   THEN: use keyword matching algorithm for extracting preferences like type=restaurant, area=north
        #   no match found? -> Use python-Levenshtein algorithm
        # ELSE:
        #       Think about what to do in nother cases like affirm(), reqmore() etc.
        
        # 3.
        # IF THERE IS SUFFICIENT INFO: Find restaurant that meets the needs

        # 4.
        # Move to new state and generate the system utterance
        output = self.new_state(input)       

        return output


    def dialog_system(self):
        current_state = "Welcome"
        print("System:  Hello, welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?")

        while current_state != "End":
            user_input = input("Me: ").lower()
            
            system_utterence = self.process_user_input(user_input)
        
            print("System: ", system_utterence)

        print("System: Bye!")

system_dialog = SystemDialog()
system_dialog.dialog_system()