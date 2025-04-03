from my_functions import dataframes, clean_text, init_tfidf, train_model, save_model


def main():

    df = dataframes()

    df['cleaned_text'] = clean_text(df['cleaned_text'])

    X_train, X_test, y_train, y_test = init_tfidf(df)

    model = train_model(X_train, y_train)

    save_model(model)