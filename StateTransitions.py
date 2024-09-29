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
        # "InformThatThereIsNoRestaurant",
        # "GiveRestaurantRecommendation",
    },
    "AskForMissingInfo": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        # "InformThatThereIsNoRestaurant",
        # "GiveRestaurantRecommendation",
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
        """
        Looks for restaurants that match the user preferences.
        If there is a restaurant found, move to GiveRestaurantRecommendation and return a string with the recommendation.
        If not found, move to InformThatThereIsNoRestaurant and return string that asks user to change requirements.
        """

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
            return "Sorry, I couldn't find a restaurant that matches your preferences. Can you change your requirements?"

    def ask_for_missing_info(state) -> str:
        """
        State becomes/stays AskForMissingInfo. 
        Returns the system utterance where it asks for the next preference that is still missing.
        """
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
    
    def deny(state, utterance): # TODO: CHANGE THIS ACCORDING TO THE NEW CONFIRM
        if ", right?" in state.last_system_utterance:
            # The pricerange was wrong
            if "Let me confirm" in state.last_system_utterance:
                state.still_needed_info.add("pricerange")
                del state.user_preferences("pricerange") 
            elif "You are looking for a " in state.last_system_utterance:
                state.still_needed_info.add("food")
                del state.user_preferences("food") 



    def confirm(state):
        state.current_state = "AskForConfirmation"
        if len(state.user_preferences) != 3:
            raise ValueError("User_preferences doesn't have 3 keys: \n ", state.user_preferences)
        
        food = state.user_preferences["food"]
        area = state.user_preferences["area"]
        pricerange = state.user_preferences["pricerange"]

        vowels = "aeiou"

        # Preferred food type
        if food == "any":
            food_text = "any"
        else:
            article = "an" if food[0] in vowels else "a"
            food_text = f"{article} {food}"

        # Preferred area
        if area == "any":
            area_text = "any area" 
        else:
            area_text = f"the {area} of town"
        
        # Preferred price range
        if pricerange == "any":
            pricerange_text = "and you don't care about the price range"
        else:
            pricerange_text = f"in the {pricerange} price range"
        

        sentence = f"Let me confirm, you are looking for {food_text} restaurant in {area_text} {pricerange}, right?"

        return sentence

    def extract_preferences(self, states, user_input):
        # Keyword matching: Check if there is a preference expressed in the user input
        for key, words in self.keywords.items():
            for word in words:
                if word in user_input:
                    # Check for ambiguity
                    if key in states.user_preferences:
                        # Remove this key from user preferences and add to the ambiguity dictionary.
                        states.user_preferences.pop(key)
                        states.still_needed_info.append(key)
                        states.ambiguity[key] = [states.user_preferences[key], word]                        
                    else:
                        states.user_preferences[key] = word
                        states.still_needed_info.remove(key)
                        print(
                            "SELF.PREFERENCE in keyword matching = ", self.user_preferences
                        )

        # Handle 'any' as a wildcard
        if "any" in user_input:
            for key in self.keywords.keys():
                if key in user_input or key in states.last_system_utterance:
                    states.user_preferences[key] = "any"
                    states.still_needed_info.remove(key)
           
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

    def ask_user_for_clarification(state):
        """
        """
        key = list(state.ambiguity.keys())[0]
        possible_values = state.ambiguity[key]
        return f"For the {key} preference, would you like {possible_values[0]} or {possible_values[1]}?"
    
    def fix_ambiguity(state, user_input):
        """
        Checks the user_input to see which value for a specific preference (key) the user chooses.
        It adds the correct preference to the state.user_preferences dictionary and removes this key from the ambiguity dictionary.
        """
        key = list(state.ambiguity.keys())[0]
        possible_values = state.ambiguity[key]
        
        for value in possible_values:
            if value in user_input:
                state.still_needed_info.remove(key)
                state.user_preferences[key] = value

        state.ambiguity.pop(key)




class States:
    def __init__(self):
        self.current_state = "Welcome"
        self.user_preferences = {}
        self.still_needed_info = {"area", "food", "pricerange"}
        self.helpers = State_Helpers()
        self.last_system_utterance = ""
        self.ambiguity = {}

    def inform(self, user_input) -> str:
        """
        Extracts the preferences form the user utterance. 
        If the system needs more information, it will ask for missing info.
        If not, it wil go to the AskForConfirmation state.
        Return system utterance
        """
        # First extract preferences from the dialog_act. If something is ambigu, it moves into the AskUserForClarification state.
        self.helpers.extract_preferences(self, user_input)

        # First fix the ambiguity of the user input
        if self.ambiguity != {}:
            self.current_state = "AskUserForClarification"
            system_utterance = self.helpers.ask_user_for_clarification(self)

        # Find a system utterance based on the preferences that are still missing
        if self.still_needed_info > 0:
            # System moves to state AskForMissingInfo
            system_utterance = self.helpers.ask_for_missing_info(self)
        else:
            # System moves to state AskForConfirmation
            system_utterance = self.helpers.confirm(self)
            # System moves to GiveRestaurantRecommendation OR InformThatThereIsNoRestaurant
            # system_utterance = self.helpers.find_restaurant(self)
        return system_utterance

    def Welcome(self, user_input, dialog_act) -> str:
        """ """
        system_utterance = self.last_system_utterance
        if dialog_act == "hello":
            # Stay in the Welcome state.
            # Ask the user for preferences.
            system_utterance = "Could you provide me more information about your preferred area, price range and food type?"
        elif dialog_act == "inform":
            system_utterance = self.inform(user_input)         
        elif dialog_act == "null":
            # System moves to State AskForMissingInfo
            system_utterance = self.helpers.ask_for_missing_info(self)
            
        self.last_system_utterance = system_utterance
        return system_utterance

    def AskForMissingInfo(self, user_input, dialog_act) -> str:
        """
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        """
        system_utterance = "Could you provide me more information about your preferred area, price range and food type?"
        if dialog_act == "hello":
            # Stay in this state
            # Give previous system utterance
            system_utterance = self.last_system_utterance
        elif dialog_act == "inform":  
            # System stays in AskForMissingInfo or goes to the AskForConfirmation state
            system_utterance = self.inform(user_input)
        elif dialog_act == "null":
            # System moves to State AskForMissingInfo. 
            system_utterance = self.helpers.ask_for_missing_info(self)
        return system_utterance

    def AskUserForClarification(self, user_input, dialog_act) -> str:
        """ 
        
        """
        system_utterance = self.last_system_utterance
        if dialog_act == "inform":
            self.helpers.fix_ambiguity(self, user_input)
            if self.ambiguity == {}:
                # Find a system utterance based on the preferences that are still missing
                if self.still_needed_info > 0:
                    # System moves to state AskForMissingInfo
                    system_utterance = self.helpers.ask_for_missing_info(self)
                else:
                    # System moves to state AskForConfirmation
                    system_utterance = self.helpers.confirm(self)
        
        return system_utterance
        # TODO: Think about what needs to happen if the dialog_act is NOT inform.            


    def AskForConfirmation(self, user_input, dialog_act):
        """ 
            "AskForMissingInfo",
            "AskUserForClarification",
            "AskForConfirmation",
            "InformThatThereIsNoRestaurant",
            "GiveRestaurantRecommendation",
        """
        system_utterance = self.last_system_utterance

        if dialog_act == "affirm":
            system_utterance = self.helpers.find_restaurant(self)

    def InformThatThereIsNoRestaurant(self, user_input, dialog_act):
        """
        
        """
        # TODO: Think about what to do here?
        # - change ALL requirements? Or just one??

    def GiveRestaurantRecommendation(self, user_input, dialog_act):
        """ 
        """

    def ProvideAlternativeSuggestion():
        """ """

    def AnswerAdditionalQuestion():
        """ """

    def ProvideContactInformation():
        """ """

    def End():
        """ """
