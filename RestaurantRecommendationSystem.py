# Restaurant recommendation system
from RandomForest import RandomForest
from RestaurantRecommendationClassification import Classification
import StateTransitions

##################################################################################################################
#############################        CHANGE THE PATH TO MATCH YOUR COMPUTER           #############################
##################################################################################################################

file_path_dialog = "C:\\Users\\certj\\OneDrive - Universiteit Utrecht\\School\Methods in AI research\\PROJECT GROUP 19\\MAIRS-group-19\\MAIRS-group-19\\dialog_acts.dat"


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

    def classify_user_input(self, user_input) -> str:
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
