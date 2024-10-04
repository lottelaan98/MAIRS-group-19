import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SVM:
    """
    Support Vector Machine classifies utterances based on dialog acts.
    """
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.dataset_without_duplicates = self.original_dataset.drop_duplicates(subset=['utterance content'])
        self.vectorizer = TfidfVectorizer()  # Adjust parameters as needed
        self.svm_classifier = None

    def find_x_and_y(self, dataset):
        x = self.vectorizer.fit_transform(dataset['utterance content'])
        y = dataset['dialog act']
        return x, y

    def train_and_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        self.svm_classifier = SVC()  # You can add parameters here for tuning
        self.svm_classifier.fit(x_train, y_train)
        y_pred = self.svm_classifier.predict(x_test)
        return y_test, y_pred

    def make_prediction(self, x_test):
        if self.svm_classifier is None:
            raise Exception("Model is not trained yet. Call train_and_test() first.")
        y_pred = self.svm_classifier.predict(x_test)
        return y_pred

    def report(self, y_test, y_pred):
        return classification_report(y_test, y_pred)

    def print_wrong_predictions(self, y_test, y_pred, dialog_act=None):
        # Convert y_test to an array (if it's a pandas Series)
        y_test = y_test.values if hasattr(y_test, 'values') else y_test

        # Iterate through predictions and compare them with true labels
        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                if dialog_act is None or y_test[i] == dialog_act:
                    original_row = self.original_dataset.iloc[i]
                    print(f"Index {i}: Original utterance: {original_row['utterance content']}")
                    print(f"True label = {y_test[i]}, Predicted label = {y_pred[i]}")
                    print("-" * 80)

    def difficult_cases(self, index):
        """
        This method dives deeper into patterns with whom the system has trouble classyfing them.
        The difficult cases are the utterances starting with and utterances, negation utterances and typos in them.
        For each of these three categories we have variant dictionary with utterances that occur in the dialog acts dataset
        with and, a negation or a typo. Besides, there is also a variant for each category that has utterances that do not
        occur in the dialog acts dataset. 
        """
        # and dictionaries
        and_utterances_not_in_data = {
            "inform": [
                "and it's an italian restaurant", "and it's moderately priced", 
                "and it has outdoor seating", "and the restaurant accepts credit cards", 
                "and the restaurant is family-friendly", "and it's open till midnight", 
                "and there's free parking", "and the ambiance is casual", 
                "and they serve vegetarian options", "and it's food is excellent"
            ],
            "request": [
                "and the cuisine", "and the pricerange", "and the reservation details", "and the date",
                "and the parking possibilities", "and the payment method", "and the opening hours",
                "and the contact number", "and the menu options", "and the dress code"  
            ],
            "thankyou": [
                "and thanks a lot", "and much appreciated", "and you're so kind", "and I appreciate it",
                "and many thanks", "and i'm grateful", "and that's very kind of you", "and thanks again",
                "and you're a lifesaver, thanks", "and that was very helpful, thank you"
            ],
            "requalts": [
                "and any other options", "and show me alternatives", "and can I see other choices", 
                "and what else is available", "and do you have other suggestions", 
                "and give me more options", "and what else do you recommend", 
                "and any more places", "and any other restaurants", "and other available spots"
            ],
            "null": [
                "and I'm not sure", "and I don't know", "and I have no idea", 
                "and no input right now", "and leave it blank", "and no preference", 
                "and skip this", "and no response", "and nothing to add", "and I can't decide"
            ],
            "affirm": [
                "and absolutely", "and definitely", "and that's exactly right", "and I agree completely",  
                "and that's correct", "and that's right", "and indeed", "and yes please",
                "and that sounds good", "and of course"
            ],
            "negate": [
                "and not really", "and not at all", "and I don't think so", 
                "and that's not right", "and I don't want that", "and that's incorrect", 
                "and no thanks", "and nope", "and not this time", "and that's not what I meant"
            ],
            "bye": [
                "and see you next time", "and have a good one", "and see you later", "and talk to you soon", 
                "and take care", "and have a great day", "and farewell", 
                "and good night", "and catch you later", "and I'll talk to you next time"
            ],
            "confirm": [
                "and is that correct", "and can you confirm", "and is that right", 
                "and is that okay", "and does that sound good", "and is that what you wanted", 
                "and are you sure", "and just to confirm", "and to clarify, is this correct", 
                "and please confirm"
            ],
            "hello": [
                "and howdy", "and hi there", "and hey there", "and good morning", 
                "and good afternoon", "and good evening", "and how's it going", 
                "and how are you", "and greetings", "and what's up"
            ],
            "repeat": [
                "and repeat that please", "and could you repeat", "and can you repeat", 
                "and previous", "and previous please", "and try again", "and pardon", "and i missed that",
                "and one more time", "and say that again"
            ],
            "ack": [
                "and splendid", "and very good", "and couldn't be better", "and can't be better",
                "and great", "and that's great", "and wonderful", "and fantastic", "and awesome", "and brilliant"
            ],
            "deny": [
                "and that's not it", "and i don't agree", "and no that's wrong", 
                "and that's not right", "and i deny that", "and that's incorrect", 
                "and that's not what i meant", "and that's not what i was thinking", 
                "and no, that's not what i want", "and i disagree"
            ],
            "restart": [
                "and let's start over", "and restart", "and reset the search", 
                "and begin again", "and let's try from the beginning", 
                "and clear everything", "and start fresh", "and start again", 
                "and reset all inputs", "and let's try again"
            ],
            "reqmore": [
                "and tell me more", "and more details please", "and can you elaborate", 
                "and give me more info", "and give me additional details", 
                "and I need more information", "and can you explain more", 
                "and any more details", "and what else can you tell me", 
                "and I'd like to know more"
            ]
        }

        # and dictionary 
        # utterances that occur in the data with and added at the front
        
        and_utterances_in_data = {
            "inform": [
                "and chinese", "and north", "and cheap restaurant in the north of town", "and italian",
                "and moderately priced restaurant", "and any", "and north part of town", "and i am looking for a cheap restaurant",
                "and it should be in the west part of town", "and im looking for a moderately priced restaurant and it should be in the east part of town"
            ],
            "request": [
                "and the address", "and the location", "and the area", "and don't care", "and address",
                "whats the phone number", "and whats the phone number", "and can i get the number",
                "and phone number", "request phone number"
            ],
            "thankyou": [
                "and very good thank you bye bye", "and okay thank you and goodbye", "and sorry about my mistakes thank you good bye",
                "and thank you good bye", "and thank you goodbye", "and okay thank you goodbye", "and thank you",
                "and thank you good b", "and okay thank", "and thank you bye"
            ],
            "reqalts": [
                "and is there anything else", "and is there anything", "and i want a different res",
                "and give me a different restaurant", "and how about portuguese", "and anything else",
                "and how about spanish", "and how about north american food", "and is there anything else", "and anything else"
            ],
            "null": [
                "and sorry", "and okay", "and duration", "and survey", "and free", "and sil", "and system saying hello welcome",
                "and is", "and can you", "and make a suggestion"
            ],
            "affirm": [
                "and yes", "and sure", "and right", "and right that serves italian food", "and correct",
                "and yeah im looking for a restaurant serving kosher food", "uh yes can i get a cheap restaurant in the west part of town",
                "yes moderate price", "uh yes mexican please", "yes please can i have the address and the price range of the restaurant please"
            ],
            "negate": [
                "and no", "and no im looking for an expensive restaurant in the east part of town", "no thank you good bye",
                "and no asian", "no im looking for a moderately priced restaurant that serves kosher food", "no cheap restaurant in the south part of town",
                "and no singaporean", "and no italian food", "and no gastropub", "and no hungarian"
            ],
            "bye": [
                "and goodbye", "and bye", "and okay thank you and good bye", "and very good thank you good bye", "and thanks good bye",
                "and okay forget it goodbye", "and okay good bye", "and good bye", "and okay good bye", "and you good bye"
            ],
            "confirm": [
                "and is there a thai restaurant in the west part of town", "and is it in the north part of town",
                "and is it moderately priced", "and is it serving vietnamese food", "and is it moderately priced",
                "and is it expensive", "and is there a vietnamese restaurant", "and is it in the center of town",
                "and does it serve chinese food", "and is that asian oriental type of food"
            ],
            "hello": [
                "and hello", "and hi", "and hi im looking for a moderately priced restaurant", "and hi im looking for a cheap restaurant in the south part of town",
                "and hi im looking for mexican food", "and hi im looking for an expensive restaurant in the south part of town",
                "and hello and welcome", "and hi im looking for a restaurant in the center that serves korean food",
                "and hello im looking for an expensive restaurant", "and hello hello and welcome to the Cambridge"
            ],
            "repeat": [
                "and go back", "and can you repeat that", "and repeat that", "and okay let me try this again", 
                "and again please", "and again", "and repeat", "and cant repeat", "and could you repeat that", 
                "and please repeat"
            ],
            "ack": [
                "and kay what is the addre", "and okay uh can i", "and breath okay how about thai food",
                "and fine", "and well take that one", "and okay uh", "and okay um", "and okay can you give me another restaurant",
                "and kay", "and kay um"
            ],
            "deny": [
                "and i dont want vietnamese food", "and no not indian scandinavian food", "and wrong",
                "and i dont want turkish", "and i dont want that", "and not european fusion",
                "and dont want that", "and can you change romanian food to something else",
                "and no i dont want chinese i want thai food", "and i dont want chinese i want thai food"
            ],
            "restart": [
                "and can we start over", "and okay start over", "and oh jesus christ start over", "and start again",
                "and start over", "and uh start over", "and reset", "and okay reset", "and let's start over", "and can you reset everything"
            ],
            "reqmore": [
                "and more", "and more details", "and more info", "and more please", "and tell me more",
                "and give more", "and more info now", "and need more", "and more options", "and more suggestions"
            ]
        }
        
        # Negation dictionaries
        # utterances that do not occur in the data with negation

        utterances_with_negation_not_in_data = {
            "inform": [
                "it's not an italian restaurant", "it's not moderately priced", 
                "it doesn't have outdoor seating", "the restaurant doesn't accept credit cards", 
                "the restaurant isn't family-friendly", "it's not open till midnight", 
                "there's no free parking", "the ambiance isn't casual", 
                "they don't serve vegetarian options", "the food isn't excellent"
            ],
            "request": [
                "what's not the cuisine", "what's not the pricerange", "there's no reservation details", 
                "there's no date", "there's no parking possibilities", "there's no payment method", 
                "there's no opening hours", "there's no contact number", "there's no menu options", 
                "there's no dress code"
            ],
            "thankyou": [
                "thanks but no thanks", "much appreciated but not necessary", 
                "you're not so kind", "i don't appreciate it", "many thanks but not really", 
                "i'm not grateful", "that's not very kind of you", "thanks again but no thanks", 
                "you're not a lifesaver, no thanks", "that wasn't helpful, no thank you"
            ],
            "requalts": [
                "there's no other options", "don't show me alternatives", "i can't see other choices", 
                "what else isn't available", "you don't have other suggestions", 
                "don't give me more options", "what else don't you recommend", 
                "no more places", "no other restaurants", "no other available spots"
            ],
            "null": [
                "i'm sure", "i know", "i have some idea", 
                "there is input right now", "don't leave it blank", "i have a preference", 
                "don't skip this", "i have a response", "i have something to add", "i can decide"
            ],
            "affirm": [
                "absolutely not", "definitely not", "that's not exactly right", 
                "i don't agree completely", "that's incorrect", "that's not right", 
                "no indeed", "no please", "that doesn't sound good", "no of course"
            ],
            "negate": [
                "really", "all right", "i think so", 
                "that's right", "i want that", "that's correct", 
                "yes thanks", "yep", "this time", "that's what i meant"
            ],
            "bye": [
                "don't see you next time", "don't have a good one", "don't see you later", 
                "don't talk to you soon", "don't take care", "don't have a great day", 
                "don't farewell", "don't have a good night", "don't catch you later", 
                "i won't talk to you next time"
            ],
            "confirm": [
                "that's not correct", "you can't confirm", "that's not right", 
                "that's not okay", "that doesn't sound good", "that's not what you wanted", 
                "are you not sure", "that's not to confirm", "to clarify, this isn't correct", 
                "don't confirm"
            ],
            "hello": [
                "howdy not", "hi there not", "hey there not", 
                "good morning not", "good afternoon not", "good evening not", 
                "how's it not going", "how aren't you", "no greetings", "what's not up"
            ],
            "repeat": [
                "don't repeat that please", "could you not repeat", "don't repeat", 
                "not previous", "don't previous please", "don't try again", 
                "no pardon", "i didn't miss that", "no one more time", 
                "don't say that again"
            ],
            "ack": [
                "not splendid", "not very good", 
                "could be better", "can be better", 
                "not great", "that's not great", "not wonderful", 
                "not fantastic", "not awesome", "not brilliant"
            ],
            "deny": [
                "that's it", "i agree", "that's right", 
                "that's right", "i agree", "that's correct", 
                "that's what i meant", "that's what i was thinking", 
                "that's what i want", "i agree"
            ],
            "restart": [
                "don't start over", "don't restart", "don't reset the search", 
                "don't begin again", "let's not try from the beginning", 
                "don't clear everything", "don't start fresh", "don't start again", 
                "don't reset all inputs", "let's not try again"
            ],
            "reqmore": [
                "don't tell me more", "no more details please", 
                "can you not elaborate", "don't give me more info", 
                "don't give me additional details", "i don't need more information", 
                "can you not explain more", "no more details", 
                "what else can't you tell me", "i wouldn't like to know more"
            ]
        }
        
        utterances_with_negation_in_data = {
            "inform": [
                "not chinese", "not north", "not a cheap restaurant in the north of town", "not italian",
                "not a moderately priced restaurant", "not any", "not in the north part of town", "i am not looking for a cheap restaurant",
                "it shouldn't be in the west part of town", "i'm not looking for a moderately priced restaurant and it shouldn't be in the east part of town"
            ],
            "request": [
                "not the address", "not the location", "not the area", "don't care", "not the address",
                "what's not the phone number", "what's not the phone number", "can i not get the number",
                "not the phone number", "don't request phone number"
            ],
            "thankyou": [
                "not very good thank you bye bye", "okay not thank you and goodbye", "sorry about my mistakes not thank you good bye",
                "not thank you good bye", "not thank you goodbye", "okay not thank you goodbye", "not thank you",
                "not thank you good b", "okay not thank", "not thank you bye"
            ],
            "reqalts": [
                "is there nothing else", "is there nothing", "i don't want a different restaurant",
                "don't give me a different restaurant", "how about not portuguese", "nothing else",
                "how about not spanish", "how about no north american food", "is there nothing else", "nothing else"
            ],
            "null": [
                "not sorry", "not okay", "not duration", "not survey", "not free", "not sil", "system not saying hello welcome",
                "not is", "can you not", "don't make a suggestion"
            ],
            "affirm": [
                "not yes", "not sure", "not right", "not right that serves italian food", "not correct",
                "yeah i'm not looking for a restaurant serving kosher food", "no, i'm not looking for a cheap restaurant in the west part of town",
                "not moderate price", "no not mexican please", "no, please don't give me the address or the price range of the restaurant"
            ],
            "negate": [
                "no", "no, i'm not looking for an expensive restaurant in the east part of town", "no thank you, not goodbye",
                "no, not asian", "no, i'm not looking for a moderately priced restaurant that serves kosher food", "no, not a cheap restaurant in the south part of town",
                "no, not singaporean", "no, not italian food", "no, not gastropub", "no, not hungarian"
            ],
            "bye": [
                "not goodbye", "not bye", "okay not thank you and not good bye", "not very good thank you not good bye", "thanks but not goodbye",
                "okay forget it not goodbye", "okay not good bye", "not good bye", "okay not good bye", "you not good bye"
            ],
            "confirm": [
                "is there not a thai restaurant in the west part of town", "is it not in the north part of town",
                "is it not moderately priced", "is it not serving vietnamese food", "is it not moderately priced",
                "is it not expensive", "is there not a vietnamese restaurant", "is it not in the center of town",
                "does it not serve chinese food", "is that not asian oriental type of food"
            ],
            "hello": [
                "not hello", "not hi", "hi, i'm not looking for a moderately priced restaurant", "hi, i'm not looking for a cheap restaurant in the south part of town",
                "hi, i'm not looking for mexican food", "hi, i'm not looking for an expensive restaurant in the south part of town",
                "hello and not welcome", "hi, i'm not looking for a restaurant in the center that serves korean food",
                "hello, i'm not looking for an expensive restaurant", "hello, hello, and not welcome to cambridge"
            ],
            "repeat": [
                "don't go back", "can't repeat that", "don't repeat that", "okay, don't let me try this again", 
                "not again please", "not again", "don't repeat", "can't repeat", "could you not repeat that", 
                "please don't repeat"
            ],
            "ack": [
                "kay, what's not the addre", "okay uh, can i not", "breath, okay, how about not thai food",
                "not fine", "well, don't take that one", "okay uh not", "okay um, not", "okay, can you not give me another restaurant",
                "not kay", "not kay um"
            ],
            "deny": [
                "i don't want vietnamese food", "no, not indian but yes scandinavian food", "not right",
                "i don't want turkish", "i don't want that", "i don't want european fusion",
                "i don't want that", "can you not change romanian food to something else",
                "no, i want chinese, not thai food", "no, i want chinese, not thai food"
            ],
            "restart": [
                "can we not start over", "okay, don't start over", "oh jesus christ, don't start over", "don't start again",
                "don't start over", "uh, don't start over", "don't reset", "okay, don't reset", "let's not start over", "can you not reset everything"
            ],
            "reqmore": [
                "no more",  "no more details", "no more info", "no more please", "don't tell me more", "don't give more",
                "no more info now", "don't need more", "no more options", "no more suggestions"
                ]
            }

        utterances_with_typos_not_in_data = {
            "inform": [
                "it's an itlaian restaurant", "it's modertely priced", 
                "it has outdor seating", "the restaurant aceepts credit cards", 
                "the restaurant is family-friemdly", "it's open till midnigt", 
                "there's free parkng", "the ambiance is casul", 
                "they serve vegetarain options", "it's food is exellent"
            ],
            "request": [
                "the cuisine", "the pricerage", "the reseration details", "the dtae",
                "the parking possibilites", "the payment methid", "the opning hours",
                "the contact numbr", "the menu optoins", "the dress cod"
            ],
            "thankyou": [
                "thanks a loat", "much apprecited", "you're so knd", "i apprciate it",
                "many tnahks", "i'm gratful", "that's very knd of you", "thanks agin",
                "you're a lifesver, thanks", "that was very hlpful, thank you"
            ],
            "requalts": [
                "any oher options", "show me alternatves", "can i see oher choices", 
                "what else is avilable", "do you have other sugestions", 
                "give me more optoins", "what else do you reccomend", 
                "any more plces", "any other resturants", "other avilable spots"
            ],
            "null": [
                "i'm not suer", "i don't knw", "i have no idae", 
                "no inpt right now", "leave it blnk", "no prefence", 
                "skip tis", "no reponse", "nothing to ad", "i can't decde"
            ],
            "affirm": [
                "absolutelye", "definietly", "that's exacly right", "i agree compltely",  
                "that's corerct", "that's rigth", "indedd", "yes plese",
                "that sounds goood", "of coure"
            ],
            "negate": [
                "not relly", "not at al", "i don't thik so", 
                "that's not rigt", "i don't want tht", "that's incrorrect", 
                "no thnks", "nop", "not ths time", "that's not what i meantt"
            ],
            "bye": [
                "see you next tme", "have a good oen", "see you leter", "talk to you snon", 
                "take cre", "have a great dy", "farewel", 
                "good nght", "catch you lter", "i'll talk to you nxt time"
            ],
            "confirm": [
                "is that correcct", "can you cnfirm", "is that rght", 
                "is that oky", "does that soud good", "is that what you wnted", 
                "are you suer", "just to cnfirm", "to clarfy, is this correect", 
                "pleas confirm"
            ],
            "hello": [
                "howy", "hi threre", "hey threre", "good mornng", 
                "good afternon", "good evenig", "how's it goig", 
                "how ar you", "gretings", "what's up"
            ],
            "repeat": [
                "repeat that plese", "could you rept", "can you rpeat", 
                "prevous", "previuos please", "try agan", "pardn", "i mised that",
                "one mor time", "say that agan"
            ],
            "ack": [
                "splendiddd", "verry good", "couldn't be bettr", "can't be beter",
                "grat", "that's gret", "wonderul", "fantatstic", "aesome", "brililant"
            ],
            "deny": [
                "that's not iit", "i don't agrree", "no that's wron", 
                "that's not ritght", "i dny that", "that's incorect", 
                "that's not what i ment", "that's not what i was tinkg", 
                "no, that's not what i wnt", "i disgree"
            ],
            "restart": [
                "let's stat over", "restartt", "reset the serach", 
                "begin agian", "let's try from the begnning", 
                "clear everthing", "strat fresh", "start agian", 
                "reset all inuts", "let's try agan"
            ],
            "reqmore": [
                "tell me mroe", "more deatils please", "can you elabrate", 
                "give me more inf", "give me additonal details", 
                "i need more infomation", "can you explin more", 
                "any more detals", "what else can you tel me", 
                "i'd like to know mre"
            ]}
        
        
        utterances_with_typos_in_data = {
            "inform": [
                "not chnese", "not nrth", "not a cheap restaurant in the nort of town", "not itlian",
                "not a moderatly priced restaurant", "not anny", "not in the norh part of town", "i am not looking for a cheap restuarant",
                "it shouldn't be in the wesst part of town", "i'm not looking for a moderatly priced restaurant and it shouldn't be in the east part of twon"
            ],
            "request": [
                "not the addres", "not the locatoin", "not the arrea", "dont care", "not the adrress",
                "what's not the phne number", "what's not the phon number", "can i not get the numbr",
                "not the phonne number", "dont request phone numbr"
            ],
            "thankyou": [
                "not very good thank yuo bye bye", "okay not thank you and goodbbye", "sorry about my mistkes not thank you good bye",
                "not thank yu good bye", "not thankyou goodbye", "okay not thank you goodby", "not thank yuo",
                "not thank you good b", "okay not thnk", "not thank yo bye"
            ],
            "reqalts": [
                "is there nothing elese", "is there nothng", "i dont want a diferent restaurant",
                "don't give me a diffrent restaurant", "how about not portugese", "nothing elese",
                "how about not spanih", "how about no north amrican food", "is there nothing elese", "nothing elsse"
            ],
            "null": [
                "not sory", "not oky", "not durtion", "not surevey", "not fere", "not sil", "system not sying hello welcome",
                "not is", "can you nnot", "don't make a sugeston"
            ],
            "affirm": [
                "not yess", "not suree", "not rigt", "not right that servs italian food", "not corect",
                "yeah i'm not loking for a restaurant serving ksoher food", "no, i'm not loking for a cheap restaurant in the wesst part of town",
                "not modrate price", "no not mexcian please", "no, please dont give me the address or the price rang of the restaurant"
            ],
            "negate": [
                "nno", "no, i'm not loking for an expensive restaurant in the east part of town", "no thank you, not goodby",
                "no, not asin", "no, i'm not looking for a moderatly priced restaurant that serves koshr food", "no, not a cheep restaurant in the south part of town",
                "no, not singporean", "no, not italin food", "no, not gastropuub", "no, not hungarain"
            ],
            "bye": [
                "not goobye", "not bie", "okay not thank you and not goodby", "not very good thank you not goodby", "thanks but not goodby",
                "okay forget it not goobye", "okay not goodbbye", "not goodby", "okay not good bye", "you not good bye"
            ],
            "confirm": [
                "is there not a thia restaurant in the west part of town", "is it not in the noth part of town",
                "is it not moderatly priced", "is it not serving vietnameese food", "is it not moderatly prced",
                "is it not expensve", "is there not a vietnamsee restaurant", "is it not in the cnter of town",
                "does it not serv chinese food", "is that not asain oriental type of food"
            ],
            "hello": [
                "not hhello", "not hhi", "hi, i'm not loking for a moderatly priced restaurant", "hi, i'm not loking for a cheap restaurant in the south part of town",
                "hi, i'm not looking for mexcian food", "hi, i'm not loking for an expensive restaurant in the south part of town",
                "hello and not welcom", "hi, i'm not loking for a restaurant in the cnter that serves korean food",
                "hello, i'm not loking for an expensive restaurant", "hello, hello, and not welcom to cambridge"
            ],
            "repeat": [
                "dont go bback", "can't repat that", "don't rept that", "okay, don't let me try this agian", 
                "not agian please", "not agin", "dont rept", "can't repet", "could you not repat that", 
                "pleas don't repat"
            ],
            "ack": [
                "kay, what's not the adderss", "okay uh, can i nnot", "breth, okay, how about not thia food",
                "not fne", "well, dont take that one", "okay uh nnot", "okay um, nnot", "okay, can you not give me anther restaurant",
                "not kayy", "not kay umm"
            ],
            "deny": [
                "i dont want vietnamesse food", "no, not indian but yes scandianvian food", "not ritght",
                "i dont want turkih", "i dont want tht", "i dont want european fuision",
                "i dont want thaat", "can you not chnge romanian food to something elese",
                "no, i wnat chinese, not thia food", "no, i wnat chinese, not thi food"
            ],
            "restart": [
                "can we not strat over", "okay, dont start ovr", "oh jesus chrsit, don't strat over", "dont start agian",
                "dont start ovr", "uh, don't strat over", "dont reest", "okay, dont reeset", "let's not strat over", "can you not reset evrything"
            ],
            "reqmore": [
                "no moer", "more detials", "more inof", "more plese", "tell me mre",
                "give mroe", "more info nwo", "need mroe", "more optons", "more sugestions"
            ]
        }

        
        set_of_dicts = [and_utterances_not_in_data,
                        and_utterances_in_data,
                        utterances_with_negation_not_in_data,
                        utterances_with_negation_in_data,
                        utterances_with_typos_not_in_data,
                        utterances_with_typos_in_data
                        ]
        
        data = [(key, phrase) for key, phrases in set_of_dicts[index].items() for phrase in phrases]
        df = pd.DataFrame(data, columns=['dialog act', 'utterance content']) 
        return df

    def perform_svm_difficult(self, index):
        df_difficult = self.difficult_cases(index)
        x, y = self.find_x_and_y(self.original_dataset)
        
        # transform the difficult cases by using the same vectorizer
        x_vec_diff = self.vectorizer.transform(df_difficult['utterance content'])
        y_vec_diff = df_difficult['dialog act']
        
        # split the original dataset into train/test
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.15, random_state=42)

        # Fit the SVM on the training dataset
        self.svm_classifier = SVC()
        self.svm_classifier.fit(x_train, y_train)

        # Make predictions on the difficult cases
        y_pred = self.make_prediction(x_vec_diff)

        # Create a report of the results
        class_report = self.report(y_vec_diff, y_pred)
        print(class_report)
