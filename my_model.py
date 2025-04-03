from my_functions import dataframes, clean_text

def main():
    df = dataframes()
    df['cleaned_text'] = df['text'].apply(clean_text)
    words = vocab(df)
    TFIDF_matrix = compute_tfidf(df, words)
    return TFIDF_matrix

def vocab(df):
    text = df['cleaned_text'].tolist()
    words = []
    for sentence in text:
        words.extend(sentence.split())
    words = list(set(words))
    return words

import numpy as np
def compute_tf(doc, words):
    term_frequency = np.zeros(len(words)) #creates list with zeroes
    words_in_doc = doc.split()
    for word in words_in_doc:
        if word in words:
            term_frequency[words.index(word)] += 1

    if len(words_in_doc) > 0:
        term_frequency = term_frequency / len(words_in_doc)

    return term_frequency

def compute_idf(df, words):
    docs = len(df['cleaned_text'])
    idf = np.zeros(len(words))
    for i, word in enumerate(words):
        #counts how many times a word appears in each document in the data
        count = sum(1 for doc in df['cleaned_text'] if word in doc)
        idf[i] = np.log((docs+1) / (count+1)) + 1
    return idf

def compute_tfidf(df, words):
    tfidf_matrix = []
    idf = compute_idf(df, words)
    for doc in df['cleaned_text']:
        term_frequency = compute_tf(doc, words)
        tfidf = term_frequency * idf
        tfidf_matrix.append(tfidf)
    return np.array(tfidf_matrix)


TFIDF_matrix = main()
print(TFIDF_matrix)









