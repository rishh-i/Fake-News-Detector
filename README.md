# Fake-News-Detector

## TF-IDF
Algorithm uses TF-IDF vectorisation (Term Frequency-Inverse Document Frequency) to differentiate between real and fake news. It evaluates the importance of a word in a document relative to a collection of documents - in this case, news articles. 

Term frequency is calculated by the number of times a word appears in a document divided by the total number of words in the document.

Inverse Document Frequency is the measure of how important a word is within the entire set of documents.
It is calculated by taking the log of the total number of documents divided by the number of documents containing the word.
If a word appears in every document, its IDF will be very small, indicating it’s common and not helpful for distinguishing between real and fake articles.

TF-IDF is the product of TF and IDF calculations.

## Loading Data
Load the csv files into separate data frames using pandas. Data frames are an organised structure with labelled rows and columns.

Add a column to the data frames called “label” which is either 0 (fake) or 1 (real). 

Then concatenate all the data into one data frame to make it easier to manage. Each datum will have a label indicating if it is real or fake.

Ignore_index = True  resets the index values to avoid duplicates.

df = df.sample(frac=1).reset_index(drop=True).sample is a method that randomly samples rows from a dataset allowing it to be shuffled. frac=1 tells pandas to return 100% of the data. Reset_index just resets the index of the data. Drop = True resets the old index rather than creating a new index column. 

Printing df.head just outputs the first five rows of the dataframe. This is just to inspect and check the contents of the data.

## Cleaning Data
Regular expressions used to replace multiple whitespaces with one whitespace. Special characters are also cleaned from the data.

This is a text processing standard and is also important for techniques like TF-IDF as they rely on the separation of words.

The text is then converted to lowercase so the same words are not tokenized separately e.g. dog and Dog. 

A new column in the data frame is created with the cleaned text.

## Text Extraction from URLs
Import request library and use .get() function to get the URL and its data from the server.

Regular expressions can be used to extract text from webpages but they do not work with all types of formatting. 

Instead, I have used a library called BeautifulSoup which uses a html parser and gets all the text in the main paragraphs of a webpage. 

Using list comprehension, I join all the paragraphs into one string as one datum.

## TF-IDF Initialisation
Imports necessary modules. TfidfVectorizer provides tools which handle text data. Train_test_split is used to split a dataset into two parts: one for training the model and one for testing the model. This helps ensure that the model is evaluated on data that it hasn't seen during training, which helps in assessing its performance in a more realistic manner.

### tfidf = TfidfVectorizer(max_features=5000, stop_words='english'): 
max_features limits the number of words to consider when transforming the text into numerical values. This speeds up processing. 

Stop_words tells the TfidfVectorizer to ignore common English words (like "the", "and", "in", etc.) This allows the machine to focus on more meaningful words in the text.

### X = tfidf.fit_transform(df['cleaned_text']).toarray(): 
Fit_transform fits the TF-IDF vectorizer on the cleaned data so the text is converted to a numerical representation. The output of this is a sparse matrix which is converted to an array as it is an efficient way of storing data using the toarray() function. The array is stored in X which is used to train the model.

### y = df['label'] 
This is the target variable which represents whether the news is real or fake 0-fake, 1-real. Y represents the output we’re trying to classify for each input in X.

### X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Train_test_split()  splits the dataset into training and testing subsets.
X contains the numerical representations of the text data (i.e., the TF-IDF scores of each word in each document).Y is the target vector (labels) containing the real/fake classification for each article

test_size=0.2: This parameter specifies the proportion of the data that will be used as the test set (here it is 20%). 80% will be used for training the model.

random_state=42: This parameter ensures that the split is reproducible. The number 42 is an arbitrary seed used to initialize the random number generator. Setting this ensures that the data is split in the same way every time the code is run.

Training Set: This part of the data is used to train the model. The model learns patterns, relationships, and correlations in the data from this subset.

Test Set: This part is kept separate from the training data. Once the model has been trained, it is evaluated using the test set. The test set allows you to check how well the model performs on unseen, new data.

## Training the Model
Logistic Regression is primarily used for binary classification tasks, where the goal is to categorize data into two classes (e.g., real vs. fake news).

It calculates a probability of the news being fake or real based on the vector quantities of the TF-IDF vectorization. 

## Saving the Model
Joblib library is used to save the model and tf-idf vectorizer. 
Saving the model allows for it to be used multiple times without it having to re-do its calculations and process the input data. This is an efficient method.

