import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

ps = PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


user_input = ""
while user_input != 'Exit':
    user_input = input("Enter a movie review sentence (Exit to end) \n")
    if user_input == 'exit' or user_input == 'Exit':
        break
    cleaned_input = clean_text(user_input)
    X_input = vectorizer.transform([cleaned_input]).toarray()
    prediction = svm_model.predict(X_input)
    if prediction == 1:
        print("The review is Positive.")
    else:
        print("The review is Negative.")
