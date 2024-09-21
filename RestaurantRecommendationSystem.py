# Restaurant recommendation system
import pandas as pd
from Baseline1 import Baseline1
from Baseline2 import Baseline2
from SVM import SVM
from RandomForest import RandomForest
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

dialog_state_dictionary = {
    "Welcome": {
        "AskForMissingInfo",
        "AskUserForClarification",
        "AskForConfirmation",
        "InformUser",
        "GiveRestaurantRecommendation"
    },
    "AskForMissingInfo": {
        
    }
}

def dialog_system():
    current_state = "Welcome"
    print("System:  Hello, welcome to the UU restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?")

    while current_state != "End":
        user_input = input("Me: ").lower()
        
      
        print("System: test", user_input)

    print("System: Bye!")

dialog_system()