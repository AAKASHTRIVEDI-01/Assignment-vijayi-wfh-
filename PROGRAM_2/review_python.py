#documentation of steps in comments
# Import necessary libraries
import pandas as pd
import numpy as np
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load spaCy English language model for NLP tasks (tokenization, lemmatization, etc.)
nlp = spacy.load("en_core_web_sm")

# Read the IMDB dataset from a CSV file into a pandas DataFrame
df = pd.read_csv(r"C:\Users\trive\Desktop\IMDB Dataset.csv")

# Function to preprocess the reviews: tokenization, lemmatization, and stopword removal
def preprocess_texts(texts):
    cleaned_texts = []  # List to store the cleaned reviews
    # Use spaCy's pipeline to process the texts in batches for efficiency
    for doc in nlp.pipe(texts, batch_size=100, disable=["parser", "ner"]):  # Disable unnecessary parts of the pipeline
        # Tokenize, lemmatize, and remove stopwords and non-alphabetical words
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        # Join tokens back into a cleaned string
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts

# Apply the preprocessing function to the review column
df['cleaned_review'] = preprocess_texts(df['review'])

# Convert sentiment labels ('positive'/'negative') into binary values (1/0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Vectorize the cleaned reviews using TF-IDF, considering unigrams and bigrams
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(df['cleaned_review'])  # Transform the reviews into TF-IDF feature matrix
y = df['sentiment']  # Target variable (sentiment labels)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a logistic regression model
model = LogisticRegression(max_iter=200)  # Increase max_iter if model doesn't converge
model.fit(X_train, y_train)  # Train the model on the training data

# Predict sentiment on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the evaluation metrics
print(f"\n✅ Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Function to predict sentiment for a new review
def predict_sentiment(review):
    # Preprocess the review before making a prediction
    processed_review = preprocess_texts([review])
    # Vectorize the preprocessed review
    review_vectorized = vectorizer.transform(processed_review)
    # Make the prediction using the trained model
    prediction = model.predict(review_vectorized)[0]
    # Map prediction to human-readable sentiment (Positive or Negative)
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return sentiment

# Sample 10 random reviews from the dataset to test the sentiment prediction
df_test = df.sample(10)  
# Apply the sentiment prediction function to each review in the sample
df_test['Predicted Sentiment'] = df_test['review'].apply(predict_sentiment)

# Print the actual vs predicted sentiment for each review in the sample
print("\n✅ Predicted Sentiments:")
for _, row in df_test.iterrows():
    print(f"\nReview: {row['review']}")
    # Print actual sentiment (human-readable) based on the original sentiment label
    print(f"Actual Sentiment: {'Positive 😊' if row['sentiment'] == 1 else 'Negative 😞'}")
    # Print predicted sentiment from the model
    print(f"Predicted Sentiment: {row['Predicted Sentiment']}")
