import streamlit
import joblib
from News_ML import soup_scraping

model = joblib.load("fake_news_model.pk1")
tfidf = joblib.load("tfidf_vectorizer.pk1")

streamlit.title("GUPTA Fake News Detection")
streamlit.write("Enter an article to validate: ")

choice = streamlit.selectbox("**Choose method of entry**", ["Select Option", "Text", "URL"])

if choice == "Text":
    user_input = streamlit.text_area("Enter plaintext below:", height = 200)
    if streamlit.button("Check News"):
        if user_input.strip():
            transformed_input = tfidf.transform([user_input]) # Transform the input using the TF-IDF vectorizer
            prediction = model.predict(transformed_input) # # Makes prediction

            # calculates the model's accuracy and displays to user
            probabilities = model.predict_proba(transformed_input)[0]
            confidence = round(max(probabilities) * 100, 2)
            streamlit.write(f"**Confidence**: {confidence}%")
            streamlit.progress(confidence / 100)

            if prediction[0] == 0:
                streamlit.error("This articles is probably **FAKE!**")
            else:
                streamlit.success("This news article is probably **Real!**")
        else:
            streamlit.warning("Please enter some text to analyze.")

elif choice == "URL":
    user_input = streamlit.text_input("Enter URL of article: ")
    if streamlit.button("Check News"):
        article_text, error = soup_scraping(user_input)
        streamlit.markdown(f"The article text you have provided can be found below: \n\n{article_text}")
        if error == "":
            transformed_input = tfidf.transform([article_text]).toarray()
            prediction = model.predict(transformed_input)

            # calculates the model's accuracy and displays to user
            probabilities = model.predict_proba(transformed_input)[0]
            confidence = round(max(probabilities) * 100, 2)
            streamlit.write(f"**Confidence**: {confidence}%")
            streamlit.progress(confidence / 100)

            if prediction[0] == 0:
                streamlit.error("This article is probably **FAKE!**")
            else:
                streamlit.success("This news article is probably **Real!**")
        else:
            streamlit.warning(f"There was an error: {error}")


        streamlit.markdown(f"[Click here to read the article]({user_input})")



