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
import random
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
        "AskForConfirmation",
    },
    "AskForMissingInfo": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
    },
    "AskUserForClarification": {
        "AskForMissingInfo"
        "AskUserForClarification",
        "AskForConfirmation"
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
    "ProvideContactInformation": {
        "AnswerAdditionalQuestion",
        "End",
    },
    "End": {},
}

class Restaurant:
    def __init__(self, name, area, pricerange, food, address, postcode, phone):
        self.name = name
        self.area = area
        self.pricerange = pricerange
        self.food = food
        self.address = address
        self.postcode = postcode
        self.phone = phone

keywords = {
        "pricerange": ["cheap", "moderate", "moderately", "expensive"],
        "area": ["north", "south", "east", "west", "centre", "center"],
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

class Helpers:
    def find_restaurant(self, state) -> str:
        """
        Looks for restaurants that match the user preferences.
        If there is a restaurant found, move to GiveRestaurantRecommendation and return a string with the recommendation.
        If not found, move to InformThatThereIsNoRestaurant and return string that asks user to change requirements.
        """
        state.found_restaurants = []

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
            for _, row in filtered_df.iterrows():
                restaurant = Restaurant(
                    name=row["restaurantname"],
                    area=row["area"],
                    pricerange=row["pricerange"],
                    food=row["food"],
                    address=row["addr"],
                    postcode=row["postcode"],
                    phone=row["phone"],
                )
                state.found_restaurants.append(restaurant)

            # Return the name of the first matching restaurant
            state.currently_selected_restaurant = state.found_restaurants[0]
            found_restaurant = state.currently_selected_restaurant
            state.current_state = "GiveRestaurantRecommendation"
            return self.sell_restaurant(found_restaurant)
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
        
    def sell_restaurant(found_restaurant):
        return f"I recommend {found_restaurant.name} in the {found_restaurant.area} area, 
                serving {found_restaurant.food} cuisine, with {found_restaurant.pricerange} prices."

    def ask_for_confirmation(state):
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

    def extract_preferences(self, state, user_input, overwrite):
        # Keyword matching: Check if there is a preference expressed in the user input
        for key, words in self.keywords.items():
            for word in words:
                if word in user_input:
                    # Check for ambiguity
                    if key in state.user_preferences and not overwrite:
                        # Remove this key from user preferences and add to the ambiguity dictionary.
                        state.user_preferences.pop(key)
                        state.still_needed_info.append(key)
                        state.ambiguity[key] = [state.user_preferences[key], word]                        
                    else:
                        state.user_preferences[key] = word
                        state.still_needed_info.remove(key)
                
        # Handle 'any' as a wildcard
        if "any" in user_input:
            for key in self.keywords.keys():
                if key in user_input or key in state.last_system_utterance:
                    # Check for ambiguity
                    if key in state.user_preferences and not overwrite:
                        # Remove this key from user preferences and add to the ambiguity dictionary.
                        state.user_preferences.pop(key)
                        state.still_needed_info.append(key)
                        state.ambiguity[key] = [state.user_preferences[key], "any"]                        
                    else:
                        state.user_preferences[key] = "any"
                        state.still_needed_info.remove(key)

    
           
        # Use Levenshtein algorithm if no matches found
        if not self.user_preferences:  
            # TODO: HIJ CHECK NU OF DICTIONARY AL EEN WAARDE HEEFT. GEEF ANDER IF-STATEMENT
            # TODO: HOUD REKENING MET
            result = self.perform_levenshtein(state, user_input)
            if result is not None:
                key, word = result
                state.user_preferences[key] = word
                state.still_needed_info.remove(key)   

    def perform_levenshtein(self, state, user_input):
        print("now in Levenshtein")
        for key, words in self.keywords.items():
            for word in words:
                if any(
                    Levenshtein.ratio(word, token) > 0.8
                    for token in user_input.split()
                ):
                    return (key, word)
        return None
                    
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
                break
    
    @staticmethod
    def provide_contact_info(restaurant):
        return f"The restaurant {restaurant.name} is on {restaurant.address} with post code {restaurant.postcode}. This is their phone number: {restaurant.phone}"
    
   
    
   


class Dialog_Acts:

    def ack(self, state):
        """
        Als het bij request info zit: geef alle informatie
        Als het bij restaurantRecommendation zit: Geef een andere recommendation
        
        """
        system_utterance = state.last_system_utterance
        if state.current_state == "ProvideContactInformation":
            system_utterance = Helpers.provide_contact_info(state.found_restaurants[0])
        elif state.current_state == "GiveRestaurantRecommendation":
            system_utterance = self.reqmore(state)
        else:
            system_utterance = "How else can I help you?"
        
        return system_utterance
    
    def affirm(self, state):
        system_utterance = state.last_system_utterance
        if state.current_state == "AskForConfirmation":
            system_utterance = Helpers.find_restaurant(state)
        elif state.current_state == "AskUserForClarification":
            # If the user confirmed the last system utterance
            words = state.last_system_utterance.split()
            if "town" in words[-1]:
                state.user_preferences["area"] = state.last_system_utterance.split()[11]
                state.still_needed_info.remove("area")
            elif "range" in words[-1]:
                state.user_preferences["pricerange"] = state.last_system_utterance.split()[11]
                state.still_needed_info.remove("pricerange")
            elif "food" in words[-1]:
                state.user_preferences["food"] = state.last_system_utterance.split()[11]
                state.still_needed_info.remove("food")

        return system_utterance
    
    def bye(self, state):
        state.current_state = "End"
        return "Goodbye! Enjoy your meal!"
    
    def confirm(self, state, user_input) -> str:
        """
        User input is a question for the user where he wants to confirm some information.
        Return a string with the answer.
        """
        if state.found_restaurants == []:
          raise ValueError("There are no found restaurants when calling the confirm() function")
        
        # Keyword matching: Check if there is a keyword the user input
        for key, words in keywords.items():
            for word in words:
                if word in user_input:
                    restaurant = state.currently_selected_restaurant
                    if key == "pricerange":
                        return f"{restaurant.name} is a nice place in the {restaurant.area} of town."
                    elif key == "food":
                        return f"{restaurant.name} is a nice place serving {restaurant.food} food."
                    else:
                        return f"{restaurant.name} is a nice place in the {restaurant.pricerange} price range."
        
        return "Could you please repeat that?"
    
    def deny(self, state, user_input) -> str:
        """
        Removes the preference that the user denies or removes all preferences when not specified which one.
        Then asks for the missing preferences.
        """
        state.current_state = "AskForMissingInfo"

        # Keyword matching: Check if there is a keyword in the user input
        for key, words in keywords.items():
            for word in words:
                if word in user_input:
                    # Remove this key from the user_preferences and ask for new one
                    del state.user_preferences[key]
                    state.still_needed_info.append(key)
                    return f"Okay. What is your preference for {key}?"

        # If there was no key word in here, remove all preferences.
        state.still_needed_info = {"area", "food", "pricerange"}
        state.user_preferences = {}
        return "Let us try again. Could you provide me more information about your preferred area, price range and food type?"

    def hello(self, state):
        if state.current_state == "Welcome":
            return "Could you provide me more information about your preferred area, price range and food type?"
        else:
            return state.last_system_utterance

    def inform(self, state, user_input) -> str:
            """
            Extracts the preferences form the user utterance. 
            If the system needs more information, it will ask for missing info.
            If not, it wil go to the AskForConfirmation state.
            Return system utterance
            """
            # First extract preferences from the dialog_act. If something is ambigu, it moves into the AskUserForClarification state.
            Helpers.extract_preferences(state, user_input, False)

            # First fix the ambiguity of the user input
            if state.ambiguity != {}:
                state.current_state = "AskUserForClarification"
                system_utterance = Helpers.ask_user_for_clarification(state)

            # Find a system utterance based on the preferences that are still missing
            if state.still_needed_info > 0:
                # System moves to state AskForMissingInfo
                system_utterance = Helpers.ask_for_missing_info(state)
            else:
                # System moves to state AskForConfirmation
                system_utterance = Helpers.ask_for_confirmation(state)

            return system_utterance
   
    def negate(self, state, user_input) -> str:
        system_utterance = state.last_system_utterance
        if state.current_state == "AskForConfirmation":
            previous_preferences = state.user_preferences
            Helpers.extract_preferences(state, user_input, True)
            if previous_preferences == state.user_preferences:
                # When the user didn't state what he wants to change, remove all preferences and ask again.
                state.user_preferences = {}
                state.still_needed_info = {"area", "food", "pricerange"}
                system_utterance = "I'm sorry. What is your preference for food, price range and area?"
            else:
                system_utterance = Helpers.ask_for_confirmation(state)
        return system_utterance

    
    def null(self, state, user_input):
        system_utterance = state.last_system_utterance
        if state.current_state == "Welcome":
            # System moves to State AskForMissingInfo
            system_utterance = Helpers.ask_for_missing_info(state)
        elif state.current_state == "AskForMissingInfo":
            result = Helpers.perform_levenshtein(state, user_input)
            if result is not None:
                state.current_state == "AskUserForClarification"
                key, word = result
                if key == "area":
                    system_utterance = f"Did you say you are looking for a restaurant in the {word} of town?"
                elif key == "pricerange":
                    system_utterance = f"Did you say you are looking for a restaurant in the {word} price range?"
                elif key == "food":
                    system_utterance = f"Did you say you are looking for a restaurant that serves {word} food?"
        elif state.current_state == "AskUserForClarification":
            Helpers.fix_ambiguity(state, user_input)
            if state.ambiguity == {}:
                # Find a system utterance based on the preferences that are still missing
                if state.still_needed_info > 0:
                    # System moves to state AskForMissingInfo
                    system_utterance = Helpers.ask_for_missing_info(state)
                else:
                    # System moves to state AskForConfirmation
                    system_utterance = Helpers.ask_for_confirmation(state)
            else:
                system_utterance = Helpers.ask_user_for_clarification(state)

        return system_utterance

    def repeat(state):
        return state.last_system_utterance

    def reqalts(self, state, user_input):
        """
        When in state InformThatThereIsNoRestaurant: The user_input will be in the form of "How about...".
        Then we run find_restaurants again with the new preference.
        Er zijn 2 soorten:
        1 vanuit InformThatThereIsNoRestaurant -> How about.... -> find_restaurants
        1 vanuit GiveRestaurantRecommendation -> Verwijder eerste uit de restaurants list en geef de nieuwe nummer 1
        """
        system_utterance = state.last_system_utterance
        if state.current_state == "InformThatThereIsNoRestaurant":
            # Adjust the preferences and create a new restaurant list that complies with the user preferences
            Helpers.extract_preferences(state, user_input, True)
            state.found_restaurants = []
            system_utterance = Helpers.find_restaurant(state)
        elif state.current_state == "GiveRestaurantRecommendation":
            # Remove the current found restaurant from the list
            del state.found_restaurants[0]
            # and provide another restaurant
            if not state.found_restaurants:
                state.current_state = "InformThatThereIsNoRestaurant"
                system_utterance = "Sorry, I couldn't find a restaurant that matches your preferences. Can you change your requirements?"
            else:
                state.currently_selected_restaurant = state.found_restaurants[0]
                system_utterance =  Helpers.sell_restaurant(state.currently_selected_restaurant)

        return system_utterance

    def reqmore(self, state):
        other_recommendations = [restaurant for restaurant in state.found_restaurants if restaurant.name != state.currently_selected_restaurant.name]
        if not other_recommendations:
            system_utterance = f"The restaurant {state.currently_selected_restaurant.name} is the only restaurant that meets your preferences."
        else:
            state.currently_selected_restaurant = random.choice(other_recommendations)
            system_utterance = Helpers.sell_restaurant(state.currently_selected_restaurant)
        return system_utterance

    def request(self, restaurant, user_input):
        output_text = ""
        if "address" in user_input:
            output_text = f"The address of {restaurant.name} is on {restaurant.address}. "
        if "post" in user_input:
            output_text = output_text + f"The post code of  {restaurant.name} is {restaurant.postcode}. "
        if "phone" in user_input:
            output_text = output_text + f"The phone number of  {restaurant.name} is {restaurant.phone}."

        return output_text  

    def restart(state):
        state.current_state = "Welcome"
        state.user_preferences = {}
        state.still_needed_info = {"area", "food", "pricerange"}
        state.helpers = Dialog_Acts()
        state.last_system_utterance = ""
        state.ambiguity = {}
        state.found_restaurants = []
        return "Okay. We start over. Welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?"
   
    def thankyou(self, state, user_input):
        if "bye" in user_input:
            state.current_state = "End"
            return "You're welcome. Good bye."
        else:
            return "You're welcome."
        

  

class State:
    def __init__(self):
        self.current_state = "Welcome"
        self.user_preferences = {}
        self.still_needed_info = {"area", "food", "pricerange"}
        self.dialog_acts = Dialog_Acts()
        self.helpers = Helpers()
        self.last_system_utterance = ""
        self.ambiguity = {}
        self.found_restaurants = []
        self.currently_selected_restaurant: Restaurant = None
