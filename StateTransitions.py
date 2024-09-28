# Restaurant recommendation system
import pandas as pd
from Baseline1 import Baseline1
from Baseline2 import Baseline2
from RestaurantRecommendationSystem import SystemDialog
from SVM import SVM
from RandomForest import RandomForest
from RestaurantRecommendationClassification import load_data
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################


file_path_restaurant = "C:\\Users\\toube\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\restaurant_info.csv"

file_path_dialog = "C:\\Users\\toube\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\dialog_acts.dat"


# This dictionary contains as keys all the possible dialog states. The values of these keys are the possible subsequent states.
dialog_state_dictionary = {
    "Welcome": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",  # In practice, never reaches AskForConfirmation from the Welcome state
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation",
    },
    "AskForMissingInfo": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation",
    },
    "AskUserForClarification": {
        "AskUserForClarification",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation",
    },
    "AskForConfirmation": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation",
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
    "End": {},
}


class State_Helpers:

    keywords = {
        "pricerange": ["cheap", "moderate", "expensive"],
        "area": ["north", "south", "east", "west", "centre"],
        "food": [
            "african",
            "asian oriental",
            "australasian",
            "bistro",
            "british",
            "catalan",
            "chinese",
            "cuban",
            "dutch",
            "english",
            "european",
            "french",
            "fusion",
            "gastropub",
            "indian",
            "international",
            "italian",
            "jamaican",
            "japanese",
            "korean",
            "lebanese",
            "mediterranean",
            "modern european",
            "moroccan",
            "north american",
            "persian",
            "polynesian",
            "portuguese",
            "romanian",
            "seafood",
            "spanish",
            "steakhouse",
            "swedish",
            "swiss",
            "thai",
            "traditional",
            "turkish",
            "tuscan",
            "vietnamese",
        ],
    }

    def find_restaurant(self, state) -> str:

        found_restaurant = None

        # Look in the CSV to find for any restaurants that may meet the criteria
        data_restaurants = pd.read_csv(file_path_restaurant)
        filtered_df = data_restaurants
        criteria = state.user_preferences
        # Loop through each criterion and apply the filter
        for key, value in criteria.items():
            if value == "any":
                continue
            filtered_df = filtered_df[filtered_df[key] == value]

        # Check if there are any matching restaurants
        if not filtered_df.empty:
            # Return the name of the first matching restaurant
            found_restaurant = filtered_df.iloc[0]
            state.current_state = "GiveRestaurantRecommendation"
            return f"I recommend {found_restaurant['restaurantname']} in the {found_restaurant['area']} area, serving {found_restaurant['food']} cuisine, with {restaurant['pricerange']} prices. The address is {restaurant['addr']}, postcode {restaurant['postcode']}. The phonenumber is {restaurant['phone']}."
        else:
            state.current_state = "InformThatThereIsNoRestaurant"
            return "Sorry, I couldn't find a restaurant that matches your self.preference. Can you change your requirements?"

    def ask_for_missing_info(state) -> str:
        state.current_state = "AskForMissingInfo"

        if state.still_needed_info[0] == "area":
            return "What part of town do you have in mind?"
        if state.still_needed_info[0] == "food":
            return "What kind of food would you like?"
        if state.still_needed_info[0] == "pricerange":
            return "Would you like something in the cheap, moderate, or expensive price range?"
        raise ValueError(
            "Something went wrong in ask_for_missing info. Still_needed_info = ",
            state.still_needed_info,
        )
        return "Sorry I can't hear you"  # TODO deze kiezen of valueError

    def confirm(type_of_preference, content):
        rest_of_sentence = ""

        if type_of_preference == "food":
            return "You are looking for a ", type_of_preference, " restaurant, right?"

        elif type_of_preference == "pricerange":
            if content == "dontcare":
                rest_of_sentence == "and you don't care about the price range, "
            else:
                rest_of_sentence == "in the ", content, " price range, "

        # elif type_of_preference == "area":

        return (
            "Let me confirm, you are looking for a restaurant ",
            rest_of_sentence,
            "right?",
        )

    def extract_preferences(self, states, user_input):
        # Keyword matching: Check if there is a preference expressed in the user input
        for key, words in self.keywords.items():
            for word in words:
                if word in user_input:
                    states.user_preferences[key] = word
                    states.still_needed_info.remove(key)
                    print(
                        "SELF.PREFERENCE in keyword matching = ", self.user_preferences
                    )

        # Handle 'any' as a wildcard
        if "any" in user_input:
            print("keywords: ", self.keywords)
            for key in self.keywords.keys():  # Loop over álle keywords
                if key not in self.user_preferences:
                    # HOUDT GEEN REKENING MET SPELFOUTEN
                    # VULT ANY IN VOOR ÁLLE MISSENDE KEYS.
                    states.user_preferences[key] = "any"
                    states.still_needed_info.remove(key)
                    print("SELF.PREFERENCE in any handle = ", self.user_preferences)

        # Use Levenshtein algorithm if no matches found
        if (
            not self.user_preferences
        ):  # TODO: HIJ CHECK NU OF DICTIONARY AL EEN WAARDE HEEFT. GEEF ANDER IF-STATEMENT
            # TODO: HOUD REKENING MET
            print("now in Levenshtein")
            for key, words in self.keywords.items():
                for word in words:
                    if any(
                        Levenshtein.ratio(word, token) > 0.8
                        for token in user_input.split()
                    ):
                        states.user_preferences[key] = word
                        print(
                            "SELF.PREFERENCE in Levenshtein = ", self.user_preferences
                        )


class States:
    def __init__(self):
        self.current_state = "Welcome"
        self.user_preferences = {}
        self.still_needed_info = {"area", "food", "pricerange"}
        self.helpers = State_Helpers()
        self.last_system_utterance = ""

    def Welcome(self, dialog_act):
        """ """
        system_utterance = ""
        if dialog_act == "hello":
            # Stay in the Welcome state.
            # Ask the user for preferences.
            system_utterance = "Could you provide me more information about your preferred area, price range and food type?"
        elif dialog_act == "inform":
            # First extract preferences from the dialog_act
            self.helpers.extract_preferences(self, dialog_act)
            # Find a system utterance based on the preferences that are still missing
            if self.still_needed_info > 0:
                # System moves to State AskForMissingInfo
                system_utterance = self.helpers.ask_for_missing_info(self)
            else:
                # System moves to GiveRestaurantRecommendation OR InformThatThereIsNoRestaurant
                system_utterance = self.helpers.find_restaurant(self)
        elif dialog_act == "null":
            # System moves to State AskForMissingInfo
            system_utterance = self.helpers.ask_for_missing_info(self)
        self.last_system_utterance = system_utterance
        return system_utterance

    def AskForMissingInfo(self, dialog_act):
        """
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformThatThereIsNoRestaurant",
        "GiveRestaurantRecommendation",
        """
        system_utterance = "Could you provide me more information about your preferred area, price range and food type?"
        if dialog_act == "hello":
            # Stay in this state
            # Give previous system utterance
            system_utterance = self.last_system_utterance
        elif dialog_act == "inform":
            # First extract preferences from the dialog_act
            self.helpers.extract_preferences(self, dialog_act)
            # Find a system utterance based on the preferences that are still missing
            if self.still_needed_info > 0:
                # System moves to State AskForMissingInfo
                system_utterance = self.helpers.ask_for_missing_info(self)
            else:
                # System moves to GiveRestaurantRecommendation OR InformThatThereIsNoRestaurant
                system_utterance = self.helpers.find_restaurant(self)
        elif dialog_act == "null":
            # System moves to State AskForMissingInfo
            system_utterance = self.helpers.ask_for_missing_info(self)
        # TODO: MOVING INTO THE CONFIRMATION STATE
        return system_utterance

    def AskUserForClarification():
        """ """

    def AskForConfirmation():
        """ """

    def InformThatThereIsNoRestaurant():
        """ """

    def GiveRestaurantRecommendation():
        """ """

    def ProvideAlternativeSuggestion():
        """ """

    def AnswerAdditionalQuestion():
        """ """

    def ProvideContactInformation():
        """ """

    def End():
        """ """
