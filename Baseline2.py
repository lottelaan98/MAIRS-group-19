import pandas as pd
from sklearn.model_selection import train_test_split

class Baseline2:
    """
    Baseline 2 classifies utterances based on previously defined key-words
    """
    df_keywords = {
        'request': ['what is', 'address', 'whats', 'phone', 'postcode', 'post code', 'price range', 'type of food', 'area'],
        'thankyou': ['thank you', 'okay'],
        'ack': ['okay','um','kay'],
        'affirm': ['yes', 'right', 'yea'],
        'bye': ['bye', 'thank you'],
        'null': ['noise', 'sil', 'unintelligible', 'cough'],
        'reqalts': ['how about', 'how', 'about', 'else', 'is there anything else', 'anything else', 'is there anything else'],        
        'inform': ['food', 'restaurant', 'town', 'i dont care', 'dont care', 'it doesnt matter', 'doesnt matter', 
                'center', 'north', 'east', 'south', 'west', 'any area', 'any price range', 'anything', 'moderate', 
                'cheap', 'expensive', 'any', 'thai', 'tailand', 'lebanese', 'italian', 'italian food', 'chinese', 
                'chinese food', 'spanish', 'spanish food', 'french', 'portuguese', 'korean', 'turkish', 'asian oriental', 
                'indian', 'vietnamese', 'british food', 'european', 'mediterranean', 'mediterranean food', 'gastropub',
                'moderately'],
        'deny': ['wrong', 'want', 'dont'],
        'negate': ['no'],    
        'repeat': ['repeat', 'back', 'again'],    
        'reqmore': ['more'],    
        'restart': ['start over', 'reset'],
        'hello': ['hi', 'hello'], 
        'confirm':['it is', 'it', 'is'], 
    }
    
    def __init__(self, dataset):
        self.dataset = dataset

    def classify(self, sentence):
        """
        Finds the dialog act of a given sentence by looping over the keywords dictionary.
        Returns the predicted dialog act.        
        """
        # Loop over our chosen keywords
        for dialog_act, keywords in self.df_keywords.items():
            for keyword in keywords:
                if keyword in sentence:
                    return dialog_act
        return 'unknown'

    def evaluate(self, dataset):
        """
        Evaluates our keyword classifier.
        Returns the ratio of correctly predicted dialog acts to the total number of dialog acts.
        """
        correct = 0

        for _, row in dataset.iterrows():
            # Get the actual dialog act and utterance content
            dialog_act = row['dialog act']
            utterance = row['utterance content']
            
            # Classify the utterance
            prediction = self.classify(utterance)            
            
            # Check if the prediction matches the actual dialog act
            if prediction == dialog_act:
                correct += 1

        return correct / len(dataset)

# Sources
# https://sparkbyexamples.com/pandas/pandas-split-column/#:~:text=In%20Pandas%2C%20the%20apply(),to%20split%20into%20two%20columns.
# https://www.geeksforgeeks.org/reading-dat-file-in-python/
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
# https://stackoverflow.com/questions/21930035/how-to-write-help-description-text-for-python-functions