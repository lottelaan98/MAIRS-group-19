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
import Levenshtein

##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################


file_path_restaurant = '/Users/youssefbenmansour/Downloads/restaurant_info.csv'

file_path_dialog = "/Users/youssefbenmansour/Downloads/dialog_acts.dat"


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
        self.preference = {}
        self.current_state = "Welcome"
        self.rf_classifier = self.train_random_forest_classifier()

    def change_state(self):
        self.current_state = "End"
       

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

    def process_user_input(self, user_input ) -> str:

         # Location of the the data file. CHANGE THIS ACCORDING TO THE PATH ON YOUR OWN COMPUTER
        df_rest = pd.read_csv(file_path_restaurant)


        # 1. Classify user input using the trained Random Forest classifier

        # 2. Extract preferences
        keywords = {
            'pricerange': ['cheap', 'moderate', 'expensive'],
            'area': ['north', 'south', 'east', 'west', 'centre'],
            'food': ["indian", "british", "chinese", "european", "italian", "asian oriental", "thai", 
                                   "spanish", "gastropub", "modern european", "seafood", "mediterranean", "portuguese", 
                                   "turkish", "international", "korean", "french", "jamaican", "persian", "japanese", 
                                   "lebanese", "australasian", "north american", "cuban", "bistro", "fusion", "african", 
                                   "polynesian", "traditional", "tuscan", "swiss", "moroccan", "vietnamese", "steakhouse", 
                                   "romanian", "catalan"],
        }
        
        # Keyword matching
        for key, words in keywords.items():
            for word in words:
                if word in user_input:
                    self.preference[key] = word

        
        # Use Levenshtein algorithm if no matches found
        if not self.preference:
            for key, words in keywords.items():
                for word in words:
                    if any(Levenshtein.ratio(word, token) > 0.8 for token in user_input.split()):
                        self.preference[key] = word

        # 3. Check if there is sufficient info to recommend a restaurant
        info_match = ['pricerange', 'area', 'food']
        match = all(info in self.preference for info in info_match)
        
        if match:
            # Find restaurant that meets the needs
            restaurant = self.find_restaurant(df_rest)
            if restaurant.any() != None:
                self.change_state()
                return f"I recommend {restaurant['restaurantname']} in the {restaurant['area']} area, serving {restaurant['food']} cuisine, with {restaurant['pricerange']} prices. The address is {restaurant['addr']}, postcode {restaurant['postcode']}. The phonenumber is {restaurant['phone']}."
            else:
                return "Sorry, I couldn't find a restaurant that matches your self.preference."
        else:
            # Ask for missing info if not enough data
            missing_info = [info for info in info_match if info not in self.preference]
            return f"Could you provide more information about {missing_info}?"

        # 4. Move to the new state and generate the system utterance
        output = self.new_state(user_input)
        return output

    def find_restaurant(self, test_T):
        data_restaurants = pd.read_csv(file_path_restaurant)
        filtered_df = data_restaurants
        criteria = self.preference
    # Loop through each criterion and apply the filter
        for key, value in criteria.items():
            filtered_df = filtered_df[filtered_df[key] == value]
    
     # Check if there are any matching restaurants
        if not filtered_df.empty:
        # Return the name of the first matching restaurant
            return filtered_df.iloc[0]
        else:
            return None

    def dialog_system(self):
        current_state = "Welcome"
        print("System:  Hello, welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?")

        while self.current_state != "End":
            user_input = input("Me: ").lower()
            
            system_utterence = self.process_user_input(user_input)
            
            print("System: ", system_utterence)
            

        print("System: Bye!")

system_dialog = SystemDialog()
system_dialog.dialog_system()
