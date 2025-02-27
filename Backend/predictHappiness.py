# This is where we are going to run the happiness file

import json
import joblib
import matplotlib.pyplot as plt

# Function to clean text (reuse the same logic as in training)
import re
from nltk.tokenize import word_tokenize

# Copied from the other python file
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)     # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text.lower()

# Load and preprocess JSON file
def load_json(file_path):
    with open(file_path, errors = "ignore") as file:
        data = json.load(file)

    messages = [m["content"] for m in data["messages"]]
    return messages

# Predict sentiment for messages
def predict_sentiment(messages, vectorizer, model):
    # Clean and preprocess messages
    cleaned_messages = [clean_text(msg) for msg in messages]
    
    # Transform messages using the vectorizer
    transformed_messages = vectorizer.transform(cleaned_messages)
    
    # Predict sentiment
    predictions = model.predict(transformed_messages)
    return predictions

# Generate graphs and insights
def generate_graphs(messages, predictions):
    # Sentiment distribution
    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count
    
    # Plot sentiment distribution
    plt.figure(figsize=(8, 6))
    plt.bar(['Negative', 'Positive'], [negative_count, positive_count], color=['red', 'green'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
    
    # Message length distribution
    message_lengths = [len(msg.split()) for msg in messages]
    plt.figure(figsize=(8, 6))
    plt.hist(message_lengths, bins=20, color='blue', alpha=0.7)
    plt.title('Message Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()

# Main function
def main():
    # File paths
    json_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\GeneralSmash.json'  # Replace with new JSON file path
    model_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\RealSentiment_model.pkl'
    vectorizer_file = r'C:\Users\ahche\OneDrive\Documents\GitHub\Happy-Or-Not-Happy\TFIDF_vecotrizer.pkl'
    
    # Load JSON data
    print("Loading JSON data")
    messages = load_json(json_file)
    print(f"Loaded {len(messages)} messages.")
    
    # Load SVM model and TF-IDF vectorizer
    print("Loading SVM model and TF-IDF vectorizer")
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    
    # Predict sentiment
    print("Predicting sentiment")
    predictions = predict_sentiment(messages, vectorizer, model)
    
    # Generate graphs and insights
    print("Generating statistical graphs")
    generate_graphs(messages, predictions)
    
    print("Analysis complete.")

if __name__ == '__main__':
    main()
