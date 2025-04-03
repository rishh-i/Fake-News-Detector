import pandas as pd
def dataframes():
    # creates dataframes for each csv file
    df_fake = pd.read_csv("Fake.csv.zip")
    df_real = pd.read_csv("True.csv.zip")
    df_fake2 = pd.read_csv("Fake2.csv.zip")
    df_real2 = pd.read_csv("True2.csv.zip")

    ## adds label 0-fake 1-real
    df_fake["label"] = 0
    df_real["label"] = 1
    df_fake2["label"] = 0
    df_real2["label"] = 1

    ## joins all the data in a random order
    df = pd.concat([df_fake, df_real, df_fake2, df_real2], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True) #This shuffles the data

    return df


import re
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    return text.lower()


import requests
from bs4 import BeautifulSoup
def soup_scraping(url):
    try:
        response = requests.get(url)

        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p') #this gets all the text in the paragraphs

        article_text = " ".join([para.get_text() for para in paragraphs])

        return article_text, ""

    except Exception as e:
        print(f"Error occurred: {e}")
        return "", e


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
def init_tfidf(df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english') # stop words ignores filler worlds like: in, a, or etc...
    X = tfidf.fit_transform(df['cleaned_text']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def output_stats(model,X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


import joblib
def save_model(model, tfidf):
    joblib.dump(model, "fake_news_model.pk1")
    joblib.dump(tfidf, 'tfidf_vectorizer.pk1')

