# Restaurant recommendation system
from RandomForest import RandomForest
from RestaurantRecommendationClassification import Classification
import StateTransitions
import nltk
from nltk.corpus import words
import Levenshtein
from StateTransitions import keywords
import difflib

# Download the words dataset if not already available
# nltk.download("words")

##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################

file_path_dialog = "/Users/youssefbenmansour/Downloads/dialog_acts.dat"
class SystemDialog:
    def __init__(self):
        self.classification = Classification(file_path_dialog)
        # Store the random_forest instance here
        self.random_forest = self.train_random_forest_classifier()
        # Access vectorizer after initialization
        self.vectorizer = self.random_forest.vectorizer
        self.state = StateTransitions.State()
        self.acts = StateTransitions.Dialog_Acts()
        self.turn_index = 0

    def train_random_forest_classifier(self):
        """
        This function is called in the beginning, in order to have a classifier to classify the user utterances in 15 different dialog acts
        """
        df = self.classification.load_data()
        print("One moment please. We are training our classifier.")
        random_forest = RandomForest(df)
        random_forest.perform_random_forest()

        return random_forest
    
    def get_preference_second(user_input):
        keys = ['touristic', 'romantic', 'children', 'assignedseats']
        result = {key: 'any' for key in keys}
        user_input = user_input.lower()

        keywords_2 = {
            'touristic':     ['touristic'],
            'romantic':      ['romantic'],
            'children':      ['children'],
            'assignedseats': ['assigned seats', 'reservation']
        }
    
        negations = ["no", "not", "don't", "do not", "without", "none"]

        input_words = user_input.split()
        for key, words in keywords_2.items():
            for word in words:
                if word in user_input:
                    word_idx = user_input.find(word)
                    negated = any(neg in user_input[:word_idx] for neg in negations)
                
                    if negated:
                        result[key] = f'not {word}'
                    else:
                        result[key] = word
                    break 
        for key, value in result.items():
            if value == 'any':
                for word in keywords_2[key]:
                    matches = difflib.get_close_matches(word, input_words, cutoff=0.8)
                    if matches:
                        result[key] = word
                        break

        if result['assignedseats'] in ['assigned seats', 'reservation']:
            result['assignedseats'] = 0
        return result

    

    def classify_user_input(self, user_input) -> str:
        
        def correct_sentence(sentence):
            corrected_sentence = []
            words = sentence.split()  # Split sentence into words

    # Iterate over each word in the sentence
            for word in words:
                corrected_word = word
                min_overall_distance = 2

        # Check the word against all categories
                for category in keywords:
                    for keyword in keywords[category]:
                # Compute Levenshtein distance between word and keyword
                        distance = Levenshtein.distance(word, keyword)
                
                # If a closer match is found, update the corrected word
                        if distance < min_overall_distance:
                            min_overall_distance = distance
                            corrected_word = keyword

                corrected_sentence.append(corrected_word)

            return " ".join(corrected_sentence)
        
        user_input = correct_sentence(user_input)
        """
        Input is a user utterance. Output is the predicted dialog act (i.e. class) of this user utterance.
        """
        # Transform input to match the trained model
        preprocessed_input = self.vectorizer.transform([user_input.lower()])
        predicted_class: str = self.random_forest.rf_classifier.predict(
            preprocessed_input
        )[0]
        return predicted_class

    def perform_dialog_act(self, predicted_class, user_input):
        print(predicted_class)
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
            return self.acts.restart(self.state)
        else:
            return self.acts.thankyou(self.state, user_input)

    def dialog_system(self):
        print(
            "System:  Hello, welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?"
        )

        self.state.last_system_utterance = "Hello, welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?"

        while self.state.current_state != "End":
            user_input = input("Me: ").lower()

            predicted_class = self.classify_user_input(user_input)
            # print("PREDICTED CLASS = ", predicted_class)

            system_utterance = self.perform_dialog_act(predicted_class, user_input)
            self.state.last_system_utterance = system_utterance

            print("System: ", system_utterance)


system_dialog = SystemDialog()
system_dialog.dialog_system()
