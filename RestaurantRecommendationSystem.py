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
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from StateTransitions import States


##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################


file_path_restaurant = "C:\\Users\\toube\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\restaurant_info.csv"

file_path_dialog = "C:\\Users\\toube\\OneDrive - Universiteit Utrecht\\School\\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\dialog_acts.dat"


# This dictionary contains as keys all the possible dialog states. The values of these keys are the possible subsequent states.
dialog_state_dictionary = {
    "Welcome": {
        "Welcome",
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
    "End": {}
}

class SystemDialog:
    def __init__(self):
        self.preference = {}
        self.current_state = "Welcome"
        self.random_forest = self.train_random_forest_classifier()  # Store the instance here
        self.vectorizer = self.random_forest.vectorizer  # Access vectorizer after initialization

    def train_random_forest_classifier(self):
        df = load_data()

        # Random Forest 
        print('One moment please...')
        random_forest = RandomForest(df)
        random_forest.perform_random_forest()
        
        return random_forest  # Return the entire RandomForest instance     

    def change_state(self):
        self.current_state = "End"

    def new_state(self, user_input) -> str:
        """
        Changes the current_state and uses the user_input. Returns the system output and the system utterance.
        """

        preprocessed_input = self.vectorizer.transform([user_input.lower()])  # Transform input to match the trained model
        predicted_class: str = self.random_forest.rf_classifier.predict(preprocessed_input)[0]  

        States.Welcome(predicted_class)

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
            'food': ["indian", "british", "dutch", "english", "chinese", "european", "italian", "asian oriental", "thai", 
                                   "spanish", "gastropub", "modern european", "seafood", "mediterranean", "portuguese", 
                                   "turkish", "international", "korean", "french", "jamaican", "persian", "japanese", 
                                   "lebanese", "australasian", "north american", "cuban", "bistro", "fusion", "african", 
                                   "polynesian", "traditional", "tuscan", "swiss", "moroccan", "vietnamese", "steakhouse", 
                                   "romanian", "catalan", "swedish"],
        }
        

         # Handle 'any' as a wildcard
        if 'any' in user_input:
            for key in keywords.keys():
                if key not in self.preference:
                    self.preference[key] = 'any'
                    print('SELF.PREFERENCE in any handle = ', self.preference)

        # Keyword matching: Check if there is a preference expressed in the user input
        for key, words in keywords.items():
            for word in words:
                if word in user_input:
                    self.preference[key] = word
                    print('SELF.PREFERENCE in keyword matching = ', self.preference)


        
        # Use Levenshtein algorithm if no matches found
        if not self.preference: # TODO: HIJ CHECK NU OF DICTIONARY AL EEN WAARDE HEEFT. GEEF ANDER IF-STATEMENT
            print('now in Levenshtein')
            for key, words in keywords.items():
                for word in words:
                    if any(Levenshtein.ratio(word, token) > 0.8 for token in user_input.split()):
                        self.preference[key] = word
                        print('SELF.PREFERENCE in Levenshtein = ', self.preference)


        # 3. Check if there is sufficient info to recommend a restaurant
        info_match = ['pricerange', 'area', 'food']
        match = all(info in self.preference for info in info_match)


        preprocessed_input = self.vectorizer.transform([user_input.lower()])  # Transform input to match the trained model
        predicted_class: str = self.random_forest.rf_classifier.predict(preprocessed_input)[0]  

        print('PREDICTED CLASS = ', predicted_class)
        if(predicted_class == 'bye' or predicted_class == 'thankyou' ):
            self.current_state = "finish"
            return
        '''
        b = user_input.lower()  # Lowercase the input text
        b = self.vectorizer.transform([b]) 
        preprocessed_instance = vectorizer.transform([user_input])  # Transform input to match the trained model
        
        # 1. Classify user input using the trained Random Forest classifier
        predicted_class = self.rf_classifier.predict(preprocessed_instance)[0]

        if(predicted_class == 'bye'):
            self.current_state = "finish"
            return
        '''
        
        if match:
            # Find restaurant that meets the needs
            restaurant = self.find_restaurant(df_rest)
            if restaurant is None:
                return "Sorry, I couldn't find a restaurant that matches your self.preference. Can you change your requirements"
            else:
                self.change_state()
                return f"I recommend {restaurant['restaurantname']} in the {restaurant['area']} area, serving {restaurant['food']} cuisine, with {restaurant['pricerange']} prices. The address is {restaurant['addr']}, postcode {restaurant['postcode']}. The phonenumber is {restaurant['phone']}."
            
        else:
            # Ask for missing info if not enough data
            missing_info = [info for info in info_match if info not in self.preference]
            #my_list = missing_info.copy()
           # text_of_missing = self.printmissing(my_list)
            result =""
            for i in range(len(missing_info)):
                if i ==0:
                    result += missing_info[i]
                else:
                    result += ", " + missing_info[i]

            print(result)
            return f"Could you provide more information about {result}?"

        # 4. Move to the new state and generate the system utterance
        output = self.new_state(user_input)
        return output

    def find_restaurant(self, test_T):
        data_restaurants = pd.read_csv(file_path_restaurant)
        filtered_df = data_restaurants
        criteria = self.preference
    # Loop through each criterion and apply the filter
        for key, value in criteria.items():
            if value == 'any':
                continue
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

        while self.current_state != "End" and self.current_state != "finish":
            user_input = input("Me: ").lower()
            
            system_utterence = self.process_user_input(user_input)
            
            print("System: ", system_utterence)
            
        while self.current_state == "End" and self.current_state != "finish" :
                user_input = input("Me: ").lower()
                system_utterence = self.process_user_input(user_input)
                
                if(self.current_state == "finish"):
                    print("System: Bye!")
                print("System: ", system_utterence)
                


system_dialog = SystemDialog()
system_dialog.dialog_system()
