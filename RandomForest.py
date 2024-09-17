import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to separate the first word from the rest of the sentence
def separate_first_word(text):
    first_space_index = text.find(' ')  
    first_word = text[:first_space_index]  
    rest_of_sentence = text[first_space_index+1:]  
    return first_word, rest_of_sentence

df = pd.read_csv('/Users/youssefbenmansour/Downloads/dialog_acts.dat')   
df.rename(columns={'inform im looking for a moderately priced restaurant that serves': 'dialog'}, inplace=True)
  
for i, row in df.iterrows():
    first, second = separate_first_word(row['dialog'])  
    df.at[i, 'dialog'] = second  
    df.at[i, 'class_label'] = first


class_labels = df['class_label'].unique()
# class_labels = ['ack', 'affirm', 'bye', 'confirm', 'deny', 'hello', 'inform', 'negate', 'null', 'repeat','reqalts','reqmore','request','restart','thankyou']

df2 = df
df2 = df2.drop_duplicates()

# Dataset with duplicates
vect = TfidfVectorizer(max_features=500)  
X = vect.fit_transform(df['dialog'])  
y = df['class_label']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("Accuracy with duplicates:", accuracy_score(y_test, y_pred))



# Dataset without duplicates
vect = TfidfVectorizer(max_features=500)  
X = vect.fit_transform(df2['dialog'])  
y = df2['class_label']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("Accuracy without duplicates:", accuracy_score(y_test, y_pred))








