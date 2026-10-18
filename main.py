from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import speech_recognition as sr
import pyttsx3 as py3

dt = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
dt = dt[[0,5]]
dt.columns= ['sentiment', 'text']
dt =dt[dt.sentiment != 2]
dt["sentiment"] = dt["sentiment"].map({0: 0, 4: 1})
def clean_text(text):
    return text.lower()

dt['clean_text'] = dt['text'].apply(clean_text)

x = dt["text"]
y = dt["sentiment"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.05, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
BernoulliNB_classifier = BernoulliNB().fit(x_train, y_train)
with sr.Microphone() as source:
    recognizer = sr.Recognizer()
    print("Speak now:")
    audio = recognizer.listen(source, timeout=5)
    try:
        string = recognizer.recognize_google(audio)
        print("You said: " + string)
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

print("positive" if BernoulliNB_classifier.predict(X=vectorizer.transform([string])) else "negative")
engine = py3.init()
engine.say("The sentiment is " + ("positive" if BernoulliNB_classifier.predict(X=vectorizer.transform([string])) else "negative"))
engine.runAndWait()
print("Accuracy is:", accuracy_score(y_test, BernoulliNB_classifier.predict(X=x_test)))
