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

def load_data() -> pd.DataFrame:
    # Location of the the data file. CHANGE THIS ACCORDING TO THE PATH ON YOUR OWN COMPUTER
    df_rest = pd.read_csv('1b_restaurant_info.csv') 
    df_rest = df_rest.astype(str)
    
    df_rest = df_rest.rename(columns={'restaurantname,"pricerange","area","food","phone","addr","postcode"': 'data'})

    # Split each row into 'restaurantname', 'pricerange', 'area', 'food', 'phone', 'address', and 'postcode'
    df_rest['restaurantname'] = df_rest['data'].apply(lambda x: x.split(',"')[0])
    df_rest['pricerange'] = df_rest['data'].apply(lambda x: x.split(',')[1])
    df_rest['area'] = df_rest['data'].apply(lambda x: x.split(',')[2])
    df_rest['food'] = df_rest['data'].apply(lambda x: x.split(',')[3])
    df_rest['phone'] = df_rest['data'].apply(lambda x: x.split(',')[4])
    df_rest['address'] = df_rest['data'].apply(lambda x: x.split(',')[5])
    df_rest['postcode'] = df_rest['data'].apply(lambda x: x.split(',')[6])
    df_rest = df_rest.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    df_rest = df_rest.drop(columns=['data'])
    return df_rest

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

    def process_user_input(self, user_input) -> str:
        # 1. Classify user input using the trained Random Forest classifier

        # 2. Extract preferences
        preferences = {}
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
                    preferences[key] = word
        
        # Use Levenshtein algorithm if no matches found
        if not preferences:
            import Levenshtein
            for key, words in keywords.items():
                for word in words:
                    if any(Levenshtein.ratio(word, token) > 0.8 for token in user_input.split()):
                        preferences[key] = word

        # 3. Check if there is sufficient info to recommend a restaurant
        info_match = ['pricerange', 'area', 'food']
        match = all(info in preferences for info in info_match)
        
        if match:
            # Find restaurant that meets the needs
            restaurant = self.find_restaurant(preferences, df_rest)
            if restaurant:
                return f"I recommend {df_rest['restaurantname']} in the {df_rest['area']} area, serving {df_rest['food']} cuisine, with {df_rest['price']} prices. The address is [df_rest['address']}, postcode {df_rest['postcode']}. The phonenumber is {df_rest['phone']}."
            else:
                return "Sorry, I couldn't find a restaurant that matches your preferences."
        else:
            # Ask for missing info if not enough data
            missing_info = [info for info in info_match if info not in preferences]
            return f"Could you provide more information about {missing_info}?"

        # 4. Move to the new state and generate the system utterance
        output = self.new_state(user_input)
        return output

    def find_restaurant(self, preferences, data_restaurants):
        
        for restaurant in data_restaurants:
            if all(preferences[key] == restaurant[key] for key in preferences):
                return restaurant
        return None

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
